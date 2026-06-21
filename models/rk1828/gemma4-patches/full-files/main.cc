// Copyright (c) 2025 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "rknn_gemma4.h"
#include "audio_utils.h"
#include "image_utils.h"

#include "Tokenizer.h"
#include "float16.h"
#include "time_utils.h"

#include <fcntl.h>
#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <iostream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <nlohmann/json.hpp>

#define LOGW(fmt, ...) printf("\033[33m" fmt "\033[0m", ##__VA_ARGS__)

static bool is_empty_arg(const char* arg)
{
  return arg == NULL || arg[0] == '\0';
}

// Non-static so rknn_gemma4.cc's RTSTREAM loop can read the first-response-token
// timestamp to report true post-stream latency.
int64_t        g_first_token_time_us = 0;
static bool    g_first_decode        = true;

// ── RK1828 AudioLLM server mode (Phase 2, opt-in via trailing "-" argv) ──
// Protocol (matches rkvoice_stream.runtime.rknn3_worker.AudioLLMWorker / the
// Python mock fixture, bit-for-bit):
//   - argv:   <model_dir> [--device-id <id>] -      (trailing "-" = server)
//   - stderr: "READY 1" once Init (LLM + audio encoder) completes (handshake +
//             AUDIO_LLM_PROTOCOL_VERSION=1).
//   - stdin:  one JSON request line per turn:
//                 {"audio_ref": "/path.wav", "prompt": "...",
//                  "max_new_tokens": 256}
//   - stdout: per generated token  [uint32 LE len][utf8 token bytes]
//             per request end       [uint32 LE 0xFFFFFFFE]  (EOS sentinel)
//   - EOF on stdin -> exit.
// Like the TTS server mode, the real stdout fd is dup'd to g_frame_fd for raw
// binary frames and the C stdout FILE* is re-pointed at stderr so stray
// printf() in this file / rknn_gemma4.cc never corrupts the token stream.
#define AUDIO_LLM_PROTOCOL_VERSION 1
static const uint32_t AUDIO_LLM_END_OF_STREAM = 0xFFFFFFFEu;
static bool g_server_mode = false;
static int  g_frame_fd    = -1;

static void frame_write_all(const void* buf, size_t len)
{
  const char* p = (const char*)buf;
  while (len > 0) {
    ssize_t n = write(g_frame_fd, p, len);
    if (n <= 0) {
      if (n < 0 && errno == EINTR) continue;
      fprintf(stderr, "[server] frame write failed (n=%zd errno=%d)\n", n, errno);
      return;
    }
    p += n;
    len -= (size_t)n;
  }
}

// Emit one UTF-8 text token as [uint32 LE len][bytes].
static void emit_token_frame(const std::string& piece)
{
  if (piece.empty()) return;  // 0-len frames carry no text; skip (reader tolerates).
  uint32_t len = (uint32_t)piece.size();
  frame_write_all(&len, sizeof(len));
  frame_write_all(piece.data(), piece.size());
}

static void emit_eos_frame()
{
  uint32_t marker = AUDIO_LLM_END_OF_STREAM;
  frame_write_all(&marker, sizeof(marker));
}

static const rknn3_sampling_params SAMPLE_PARAMS = {
    1,    // top_k
    0.9f, // top_p
    1.0f, // temperature
    1.0f, // repeat_penalty
    0.0f, // frequency_penalty
    0.0f  // presence_penalty
};

static int argmax(const float16* data, int size)
{
  if (!data || size <= 0) {
    return -1;
  }

  int max_id = 0;
  for (int i = 1; i < size; ++i) {
    if (fp16_to_fp32(data[i]) > fp16_to_fp32(data[max_id])) {
      max_id = i;
    }
  }
  return max_id;
}

static int output_callback(void* userdata, rknn3_tensor* output_tensors, uint32_t n_output_tensors,
                           LLMOutputCallbackState state)
{
  (void)userdata;
  (void)output_tensors;
  (void)n_output_tensors;
  (void)state;
  return 0;
}

static int sampling_callback(void* userdata, float16* logits, char* logits_name)
{
  (void)logits_name;
  embedding_info* embed_info = (embedding_info*)userdata;
  return argmax(logits, embed_info->vocab_size);
}

static int result_callback(void* userdata, RKLLMResult* result, LLMCallState state)
{
  Tokenizer* tokenizer = (Tokenizer*)userdata;

  if (state == RKLLM_RUN_ERROR) {
    printf("\n\nError occurred during inference\n");
    fflush(stdout);
    return 0;
  }

  if (state == RKLLM_RUN_FINISH || state == RKLLM_RUN_WAITING || state == RKLLM_RUN_MAX_NEW_TOKEN_REACHED ||
      state == RKLLM_RUN_STOP) {
    fflush(stdout);
    return 0;
  }

  if (state == RKLLM_RUN_NORMAL) {
    std::string piece;
    if (result->num_tokens == 1) {
      piece = tokenizer->TokenToPiece(result->token_ids[0]);
    } else {
      piece = tokenizer->Decode(result->token_ids, result->num_tokens);
    }

    if (g_server_mode) {
      // Server mode: emit the token as a length-prefixed UTF-8 frame on the raw
      // (dup'd) stdout fd; the C stdout FILE* points at stderr here.
      emit_token_frame(piece);
    } else {
      printf("%s", piece.c_str());
    }
    if (g_first_decode) {
      g_first_token_time_us = getCurrentTimeUs();
      g_first_decode        = false;
    }
    fflush(stdout);
  }

  return 0;
}

static bool is_image_probe_text(const char* text, int32_t text_len) {
  std::string s(text, text_len);
  return s == "<|image>" ||
         s == "<|image|>" ||
         s == "<image|>" ||
         s == "<|image><|image|>";
}

static int tokenizer_callback(void* userdata, const char* text, int32_t text_len, int32_t* tokens,
                              int32_t n_tokens_max)
{
  Tokenizer* tokenizer = (Tokenizer*)userdata;
  int n_tokens = tokenizer->Tokenize(text, text_len, tokens, n_tokens_max);

  if (n_tokens <= 0) {
    printf("tokenizer failed for %s\n", text);
    return n_tokens;
  }

  const int bos_id = 2; //防止在Gemma Tokenize 在查询图像占位符时，意外混入多余的 BOS Token
  if (is_image_probe_text(text, text_len) && n_tokens > 0 && tokens[0] == bos_id) {
    if (n_tokens > 1) {
      std::memmove(tokens, tokens + 1, (n_tokens - 1) * sizeof(int32_t));
    }
    --n_tokens;
  }

  return n_tokens;
}

static int embed_callback(void* userdata, int32_t* tokens, uint64_t num_tokens, void* embed, uint64_t len)
{
  embedding_info* embed_info = (embedding_info*)userdata;
  uint64_t        row_size   = (uint64_t)embed_info->embedding_dim * sizeof(float16);

  if (len != num_tokens * row_size) {
    printf("invalid embed buffer\n");
    return -1;
  }

  for (uint64_t i = 0; i < num_tokens; ++i) {
    int32_t token_id = tokens[i];
    if (token_id < 0 || token_id >= embed_info->vocab_size) {
      printf("invalid token id: %d\n", token_id);
      return -1;
    }
    memcpy((unsigned char*)embed + i * row_size,
           embed_info->embedding_data + (uint64_t)token_id * embed_info->embedding_dim, row_size);
  }

  return 0;
}

static void dump_tensor_attr(rknn3_tensor_attr* attrs)
{
  std::string shape_str = "";
  for (uint32_t j = 0; j < attrs->n_dims; j++) {
    shape_str += std::to_string(attrs->shape[j]);
    if (j < attrs->n_dims - 1) {
      shape_str += ", ";
    }
  }

  std::string stride_str = "";
  for (uint32_t j = 0; j < attrs->n_stride; j++) {
    stride_str += std::to_string(attrs->stride[j]);
    if (j < attrs->n_stride - 1) {
      stride_str += ", ";
    }
  }

  printf("  name=%s,core_id=%d, n_dims=%d, shape=[%s], stride=[%s], aligned_size=%ld, layout=%s, dtype=%s, qnt_type=%s, scale=%f, zero_point=%d\n",
                    attrs->name, attrs->core_id, attrs->n_dims, shape_str.c_str(), stride_str.c_str(), attrs->aligned_size,
                    rknn3_get_layout_string(attrs->layout), rknn3_get_type_string(attrs->dtype),
                    rknn3_get_qnt_type_string(attrs->qnt_type), attrs->qnt_info.scale, attrs->qnt_info.zero_point);
}

/* Returns element size in bytes for an rknn3_tensor_type integer value. */
static size_t get_dtype_elem_size(int dtype)
{
  switch (dtype) {
  case 0:  return 4;   /* FLOAT32   */
  case 1:  return 2;   /* FLOAT16   */
  case 2:  return 1;   /* INT8      */
  case 3:  return 1;   /* UINT8     */
  case 4:  return 2;   /* INT16     */
  case 5:  return 2;   /* UINT16    */
  case 6:  return 4;   /* INT32     */
  case 7:  return 4;   /* UINT32    */
  case 8:  return 8;   /* INT64     */
  case 9:  return 8;   /* UINT64    */
  case 10: return 1;   /* BOOL      */
  case 11: return 1;   /* INT4      */
  case 12: return 1;   /* FLOAT8E4M3FN */
  case 13: return 2;   /* BFLOAT16  */
  case 14: return 1;   /* FLOAT8E8M0   */
  case 15: return 1;   /* FLOAT4E2M1   */
  default: return 1;
  }
}

static int input_callback(void* userdata, rknn3_tensor* input_tensors, uint32_t n_input_tensors,
                          LLMInputCallbackParam param)
{
  input_cb_userdata* cb_data   = (input_cb_userdata*)userdata;
  embedding_info*    embed_info = &cb_data->per_layer_embed;
  uint64_t           row_size   = (uint64_t)embed_info->embedding_dim * sizeof(float16);

  for (uint32_t i = 0; i < n_input_tensors; ++i) {
    // Handle rope cache tensors (NC1HWC2): copy slice [:,:,:,param.pos:param.pos+num_tokens,:]
    bool handled_as_rope = false;
    for (int c = 0; c < 4; c++) {
      if (strcmp(input_tensors[i].attr->name, ROPE_CACHE_NAMES[c]) == 0) {
        const rope_cache_tensor* cache     = &cb_data->rope_caches[c];
        const size_t             elem_sz   = get_dtype_elem_size(cache->dtype);
        const int                C1        = cache->shape[1];
        const size_t             c2_bytes  = (size_t)cache->shape[4] * elem_sz;
        const size_t             src_stride = (size_t)cache->shape[3] * c2_bytes; /* W * C2 * elem_sz */
        const size_t             dst_stride = (size_t)input_tensors[i].attr->shape[3] * c2_bytes;
        const uint8_t*           src = (const uint8_t*)cache->data
                                       + (size_t)param.pos * c2_bytes;
        uint8_t*                 dst = (uint8_t*)input_tensors[i].mem->virt_addr;
        for (int c1 = 0; c1 < C1; c1++, src += src_stride, dst += dst_stride) {
          memcpy(dst, src, dst_stride);
        }
        handled_as_rope = true;
        break;
      }
    }
    if (handled_as_rope) continue;

    if (strcmp(input_tensors[i].attr->name, "per_layer_inputs") != 0) {
      continue;
    }

    float16* dst = (float16*)input_tensors[i].mem->virt_addr;
    for (int t = 0; t < param.num_tokens; ++t) {
      int32_t token_id = param.tokens[t];

      // Handle special audio and image token (<|audio|> and <|image|>) by mapping it to [PAD] token.
      // This is same to python implementation
      if (token_id == 258881 || token_id == 258880) { 
        token_id = 0; // map to [PAD] token
      }

      if (token_id >= 0 && token_id < embed_info->vocab_size) {
        memcpy(dst + (uint64_t)t * embed_info->embedding_dim,
               embed_info->embedding_data + (uint64_t)token_id * embed_info->embedding_dim, row_size);
      } else {
        printf("Warning: token_id %d out of vocab range, filling with zeros\n", token_id);
        memset(dst + (uint64_t)t * embed_info->embedding_dim, 0, row_size);
      }
    }
  }

  return 0;
}

static int load_embedding(const char* path, int vocab_size, embedding_info* embed_info)
{
  struct stat st;
  memset(embed_info, 0, sizeof(*embed_info));
  embed_info->fd = -1;

  embed_info->fd = open(path, O_RDONLY);
  if (embed_info->fd == -1) {
    printf("Failed to open embedding file: %s\n", path);
    return -1;
  }

  if (fstat(embed_info->fd, &st) == -1) {
    printf("Failed to get embedding file size: %s\n", path);
    close(embed_info->fd);
    embed_info->fd = -1;
    return -1;
  }

  embed_info->embedding_data = (float16*)mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, embed_info->fd, 0);
  if (embed_info->embedding_data == MAP_FAILED) {
    printf("Failed to mmap embedding file: %s\n", path);
    close(embed_info->fd);
    embed_info->fd             = -1;
    embed_info->embedding_data = NULL;
    return -1;
  }

  embed_info->vocab_size    = vocab_size;
  embed_info->embedding_dim = (int)((st.st_size / (uint64_t)vocab_size) / sizeof(float16));
  embed_info->data_size     = (size_t)st.st_size;
  return 0;
}

static void release_embedding(embedding_info* embed_info)
{
  if (!embed_info) {
    return;
  }

  if (embed_info->embedding_data) {
    munmap(embed_info->embedding_data, embed_info->data_size);
    embed_info->embedding_data = NULL;
  }

  if (embed_info->fd != -1) {
    close(embed_info->fd);
    embed_info->fd = -1;
  }
}

/* safetensors loader — uses nlohmann::json for header parsing. */
static int load_safetensors(const char* path, rope_cache_tensor caches[4],
                            int* fd_out, void** mmap_base_out, size_t* mmap_size_out)
{
  int         fd          = -1;
  void*       map         = MAP_FAILED;
  uint64_t    header_size = 0;
  struct stat st;
  int         ret         = -1;

  fd = open(path, O_RDONLY);
  if (fd < 0) {
    printf("Failed to open safetensors file: %s\n", path);
    goto err;
  }
  if (fstat(fd, &st) < 0) {
    printf("Failed to stat safetensors file: %s\n", path);
    goto err;
  }

  /* Read the 8-byte little-endian header size */
  if (read(fd, &header_size, 8) != 8) {
    printf("Failed to read safetensors header size\n");
    goto err;
  }
  if (header_size == 0 || header_size > (uint64_t)st.st_size - 8) {
    printf("Invalid safetensors header size: %" PRIu64 "\n", header_size);
    goto err;
  }

  /* mmap the whole file; JSON header starts at byte 8, tensor data after that */
  map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (map == MAP_FAILED) {
    printf("Failed to mmap safetensors file: %s\n", path);
    goto err;
  }

  /* Parse JSON header directly from the mmap'd region */
  {
    const char*    json_ptr  = (const char*)map + 8;
    const uint8_t* data_base = (const uint8_t*)map + 8 + header_size;
    try {
      nlohmann::json j = nlohmann::json::parse(json_ptr, json_ptr + header_size);

      // dtype/layout are stored as integers inside __metadata__.index (a JSON string).
      nlohmann::json meta_index = nlohmann::json::parse(
          j.at("__metadata__").at("index").get<std::string>());

      ret = 0;
      for (int i = 0; i < 4; i++) {
        const auto& meta_t = meta_index.at(ROPE_CACHE_NAMES[i]);
        int         dtype  = meta_t.at("dtype").get<int>();
        int         layout = meta_t.at("layout").get<int>();

        const auto& t      = j.at(ROPE_CACHE_NAMES[i]);
        auto shape_v   = t.at("shape").get<std::vector<int>>();
        auto offsets_v = t.at("data_offsets").get<std::vector<int64_t>>();

        int n_dims = (int)shape_v.size();
        if (n_dims != 5 || layout != 3 /* RKNN3_TENSOR_NC1HWC2 */) {
          printf("Tensor '%s': expected 5-D NC1HWC2 (layout=%d, n_dims=%d)\n",
                 ROPE_CACHE_NAMES[i], layout, n_dims);
          ret = -1;
          break;
        }
        caches[i].data   = (void*)(data_base + offsets_v[0]);
        caches[i].n_dims = n_dims;
        caches[i].dtype  = dtype;
        caches[i].layout = layout;
        for (int d = 0; d < n_dims; d++) caches[i].shape[d] = shape_v[d];
        printf("Loaded %-24s  dtype=%-2d  shape=[%d,%d,%d,%d,%d]\n",
               ROPE_CACHE_NAMES[i], dtype,
               caches[i].shape[0], caches[i].shape[1], caches[i].shape[2],
               caches[i].shape[3], caches[i].shape[4]);
      }
    } catch (const nlohmann::json::exception& e) {
      printf("Failed to parse safetensors JSON: %s\n", e.what());
      ret = -1;
    }
  }

err:
  if (ret != 0) {
    if (map != MAP_FAILED) munmap(map, (size_t)st.st_size);
    if (fd >= 0) close(fd);
    return ret;
  }
  *fd_out        = fd;
  *mmap_base_out = map;
  *mmap_size_out = (size_t)st.st_size;
  return 0;
}

static void release_safetensors(input_cb_userdata* cb_data)
{
  if (!cb_data) return;
  if (cb_data->rope_mmap_base && cb_data->rope_mmap_base != MAP_FAILED) {
    munmap(cb_data->rope_mmap_base, cb_data->rope_mmap_size);
    cb_data->rope_mmap_base = NULL;
  }
  if (cb_data->rope_fd >= 0) {
    close(cb_data->rope_fd);
    cb_data->rope_fd = -1;
  }
}

static void print_usage(const char* program)
{
  LOGW("Usage fixed all args:\n");
  LOGW("  %s <llm_model_path> <llm_weight_path> <llm_core_mask> "
       "<tokenizer_path> <embedding_path> <max_context_len> <max_new_tokens> "
       "<per_layer_embed_path> <safetensors_path> "
       "<audio_model_path> <audio_weight_path> <audio_core_mask> "
       "<vision_model_path> <vision_weight_path> <vision_core_mask> "
       "<audio_path> <image_path> [prompt]\n",
       program);
  LOGW("\nNote:\n");
  LOGW("  4 modes: LLM only | LLM+Audio | LLM+Vision | LLM+Audio+Vision\n");
  LOGW("  Use empty string \"\" for both rknn and weight of a modality to skip loading it.\n");
  LOGW("  Example LLM only: %s <llm params>... \"\" \"\" 0 \"\" \"\" 0 \"\" \"\" [prompt]\n", program);
  LOGW("  Example LLM+Audio: %s <llm params>... audio.rknn audio.weight 0xff \"\" \"\" 0 audio.wav \"\" \"<audio>将语音转为文本\"\n", program);
  LOGW("  Example LLM+Vision: %s <llm params>... \"\" \"\" 0 vis.rknn vis.weight 0xff \"\" img.jpg \"<image>请描述图片\"\n", program);
  LOGW("\nExample LLM+Audio+Vision: %s "
       "gemma-4-e2b-it.rknn gemma-4-e2b-it.weight 0xff "
       "gemma-4-e2b-it.tokenizer.gguf gemma-4-e2b-it.embed.bin 16384 2048 "
       "gemma-4-e2b-it_per_layer_inputs.embed.bin gemma-4-e2b-it.safetensors "
       "gemma-4-e2b-it-audio.rknn gemma-4-e2b-it-audio.weight 0xff "
       "gemma-4-vision.rknn gemma-4-vision.weight 0xff "
       "demo_16k_mono_f32.wav test.jpg \"<image>请描述图片\"\n",
       program);
}

static int get_vision_n_tokens_main(rknn_gemma4_vision_context* vision_ctx)
{
  if (!vision_ctx || !vision_ctx->embeds_shape || vision_ctx->embeds_ndims == 0) return 0;
  if (vision_ctx->embeds_ndims >= 3) return (int)vision_ctx->embeds_shape[vision_ctx->embeds_ndims - 2];
  if (vision_ctx->embeds_ndims == 2) return (int)vision_ctx->embeds_shape[0];
  return 0;
}


static void print_vocab_info(const VocabInfo* vocab_info)
{
  printf("vocab_info: vocab_size=%d, special_bos_id=[", vocab_info->vocab_size);
  for (int i = 0; i < vocab_info->n_special_bos_id; ++i) {
    printf("%d%s", vocab_info->special_bos_id[i], (i + 1 < vocab_info->n_special_bos_id) ? ", " : "");
  }
  printf("], special_eos_id=[");
  for (int i = 0; i < vocab_info->n_special_eos_id; ++i) {
    printf("%d%s", vocab_info->special_eos_id[i], (i + 1 < vocab_info->n_special_eos_id) ? ", " : "");
  }
  printf("]\n");
}

static void print_llm_config(const rknn3_llm_config* config, int32_t max_new_tokens)
{
  printf("\n");
  printf("=============================================================\n");
  printf("%-32s: %-8d\n", "Max Context Length", config->max_ctx_len);
  printf("%-32s: %-8d\n", "Max Position Embeddings", config->max_position_embeddings);
  printf("%-32s: %s\n", "Model Type", config->model_type);
  printf("%-32s: %s\n", "Task Type",
         config->task_type == RKNN3_LLM_TASK_GENERATE ? "RKNN3_LLM_TASK_GENERATE" : "RKNN3_LLM_TASK_EMBEDDING");
  printf("%-32s: %-8d\n", "Max New Tokens", max_new_tokens);
  printf("=============================================================\n\n");
}

static void print_perf(const rknn_perf_metrics_t* perf)
{
  float prefill_us       = (float)(g_first_token_time_us - perf->llm_start_time);
  float prefill_ms       = prefill_us / 1000.0f;
  float prefill_s        = prefill_us / 1000000.0f;
  int   prefill_n_tokens = perf->n_prefill_tokens;
  float prefill_tpt      = prefill_n_tokens == 0 ? 0.0f : prefill_ms / prefill_n_tokens;
  float prefill_tps      = prefill_n_tokens == 0 || prefill_s == 0.0f ? 0.0f : prefill_n_tokens / prefill_s;

  float decode_us       = (float)(perf->llm_end_time - g_first_token_time_us);
  float decode_ms       = decode_us / 1000.0f;
  float decode_s        = decode_us / 1000000.0f;
  int   decode_n_tokens = perf->n_decode_tokens;
  float decode_tpt      = decode_n_tokens == 0 ? 0.0f : decode_ms / decode_n_tokens;
  float decode_tps      = decode_n_tokens == 0 || decode_s == 0.0f ? 0.0f : decode_n_tokens / decode_s;

  printf("\n-----------------------------------------------------------------------------------------\n");
  printf(" %-10s | %-16s | %-8s | %-20s | %-20s \n", "Stage", "Total Time (ms)", "Tokens",
         "Time per Token (ms)", "Tokens per Second");
  printf("-----------------------------------------------------------------------------------------\n");
  printf(" %-10s | %-16.2f | %-8d | %-20.2f | %-20.2f \n", "Prefill", prefill_ms, prefill_n_tokens,
         prefill_tpt, prefill_tps);
  printf(" %-10s | %-16.2f | %-8d | %-20.2f | %-20.2f \n", "Generate", decode_ms, decode_n_tokens,
         decode_tpt, decode_tps);
  printf("-----------------------------------------------------------------------------------------\n");

  if (perf->audio_latency > 0) {
    printf(" Audio latency = %.2f ms, FPS = %.2f\n",
           (float)perf->audio_latency / 1000.f, 1000.f * 1000.f / (float)perf->audio_latency);
  }


  if (perf->vision_latency > 0) {
    printf(" Vision latency = %.2f ms, FPS = %.2f\n",
           (int64_t)perf->vision_latency / 1000.f, 1000.f * 1000.f / (int64_t)perf->vision_latency);
  }
}

// ── server mode helpers ──────────────────────────────────────────

static std::string join_path(const std::string& dir, const char* name)
{
  if (dir.empty()) return std::string(name);
  if (dir.back() == '/') return dir + name;
  return dir + "/" + name;
}

// Run one request inside the persistent server. The LLM + audio encoder are
// already Init'd; this only reads the audio, sizes the embed buffer, runs
// inference (tokens stream out via the server result_callback), then frees the
// per-request audio. Returns 0 on success (an EOS frame is always emitted by
// the caller regardless of rc so the client unblocks).
static int run_server_request(rknn_gemma4_app_context* app_ctx, bool enable_audio,
                              const std::string& audio_ref, const std::string& prompt_str,
                              int32_t max_new_tokens)
{
  audio_buffer_t src_audio;
  image_buffer_t src_image;
  float16*       audio_embeds = NULL;
  size_t         n_audio_tokens = 0;
  size_t         n_embed_audio  = 0;
  int            n_audio = 0;
  int            ret = 0;
  rknn3_llm_multimodal_tensor tensor;
  rknn_perf_metrics_t         perf;

  memset(&src_audio, 0, sizeof(audio_buffer_t));
  memset(&src_image, 0, sizeof(image_buffer_t));
  memset(&tensor, 0, sizeof(rknn3_llm_multimodal_tensor));
  memset(&perf, 0, sizeof(perf));

  if (prompt_str.empty()) {
    fprintf(stderr, "[server] empty prompt for request\n");
    return -1;
  }
  const char* prompt = prompt_str.c_str();

  // Read + size audio (mirrors main()'s long-audio chunking buffer sizing).
  if (enable_audio && !audio_ref.empty()) {
    fprintf(stderr, "[server] reading audio: %s\n", audio_ref.c_str());
    ret = read_audio(audio_ref.c_str(), &src_audio);
    if (ret != 0) {
      fprintf(stderr, "[server] read_audio fail ret=%d audio=%s\n", ret, audio_ref.c_str());
      goto req_out;
    }
    fprintf(stderr, "[server] audio: num_frames=%d channels=%d rate=%d\n",
            src_audio.num_frames, src_audio.num_channels, src_audio.sample_rate);
    {
      int n_chunks = gemma4_audio_num_chunks(src_audio.num_frames);
      if (n_chunks < 1) n_chunks = 1;
      int per_chunk_tokens = 0;
      for (int s = 0; s < app_ctx->audio.n_shapes; s++) {
        if (app_ctx->audio.embeds_dim0[s] > per_chunk_tokens) {
          per_chunk_tokens = app_ctx->audio.embeds_dim0[s];
        }
      }
      if (per_chunk_tokens <= 0) {
        per_chunk_tokens = get_n_audio(&app_ctx->audio, src_audio.num_frames);
      }
      n_audio_tokens = (size_t)n_chunks * (size_t)per_chunk_tokens;
    }
    n_embed_audio = n_audio_tokens * app_ctx->audio.embeds_dim1;
    audio_embeds = (float16*)malloc(n_embed_audio * sizeof(float16));
    if (!audio_embeds) {
      fprintf(stderr, "[server] failed to alloc audio_embeds\n");
      ret = -1;
      goto req_out;
    }
  }

  if (enable_audio &&
      (strstr(prompt, "<audio>") || strstr(prompt, "<|audio>") ||
       strstr(prompt, "<|audio|>") || strstr(prompt, "<audio|>"))) {
    n_audio = 1;
  }

  tensor.name            = "input_embeds";
  tensor.prompt          = prompt;
  tensor.enable_thinking = false;
  if (enable_audio) {
    tensor.audio.audio_embed    = audio_embeds;
    tensor.audio.n_audio_tokens = 0;  // updated after audio encoder inference
    tensor.audio.n_audio        = n_audio;
    tensor.audio.audio_start    = "<|audio>";
    tensor.audio.audio_end      = "<audio|>";
    tensor.audio.audio_content  = "<|audio|>";
  }

  g_first_decode        = true;
  g_first_token_time_us = 0;
  fprintf(stderr, "[server] --> inference (max_new_tokens=%d)\n", max_new_tokens);
  ret = inference_gemma4_model(app_ctx, tensor, &src_audio, &src_image,
                               audio_embeds, NULL, max_new_tokens, &perf);
  if (ret != RKNN3_SUCCESS) {
    fprintf(stderr, "[server] inference_gemma4_model fail ret=%d\n", ret);
    goto req_out;
  }

req_out:
  if (audio_embeds) free(audio_embeds);
  if (src_audio.data) free(src_audio.data);
  return ret;
}

// Server mode: Init LLM + audio encoder ONCE, handshake "READY 1" on stderr,
// then loop over JSON request lines emitting framed UTF-8 tokens + an EOS
// sentinel per request. model_dir holds the standard gemma-4-e2b-it.* files.
static int run_server(const std::string& model_dir, const std::string& device_id)
{
  (void)device_id;  // device selection is via env/PCIe enumeration in the runtime.

  // Resolve the standard model files inside model_dir (on-device naming).
  std::string llm_model       = join_path(model_dir, "gemma-4-e2b-it.rknn");
  std::string llm_weight      = join_path(model_dir, "gemma-4-e2b-it.weight");
  std::string tokenizer_path_s = join_path(model_dir, "gemma-4-e2b-it.tokenizer.gguf");
  std::string embedding_path_s = join_path(model_dir, "gemma-4-e2b-it.embed.bin");
  std::string per_layer_s     = join_path(model_dir, "gemma-4-e2b-it_per_layer_inputs.embed.bin");
  std::string safetensors_s   = join_path(model_dir, "gemma-4-e2b-it.safetensors");
  std::string audio_model_s   = join_path(model_dir, "gemma-4-e2b-it-audio.rknn");
  std::string audio_weight_s  = join_path(model_dir, "gemma-4-e2b-it-audio.weight");

  // The two gemma4 sub-models want DIFFERENT core masks on the RK1828 EP: the
  // LLM is built for the full 8-core layout (0xff) while the audio encoder is
  // built for the 4-core layout (0xf). They are init'd independently
  // (init_gemma4_llm vs init_gemma4_audio), so feeding them the same mask makes
  // one of the two reject ("core_mask ... is not match with npu core number").
  // Verified on RK1828 (0001:11:00.0): LLM=0xff + audio=0xf -> READY + intelligible
  // ZH output. Env vars still override for other core counts / debugging.
  uint32_t llm_core_mask   = 0xff;
  uint32_t audio_core_mask = 0xf;
  if (const char* e = getenv("GEMMA4_LLM_CORE_MASK"))   llm_core_mask   = (uint32_t)strtoul(e, NULL, 16);
  if (const char* e = getenv("GEMMA4_AUDIO_CORE_MASK")) audio_core_mask = (uint32_t)strtoul(e, NULL, 16);
  const int32_t  max_context_len = 16384;
  const bool     enable_audio    = true;
  const bool     enable_vision   = false;

  int                     ret = 0;
  VocabInfo               vocab_info;
  Tokenizer*              tokenizer = NULL;
  embedding_info          token_embedding;
  input_cb_userdata       input_cb_data;
  rknn3_llm_param         params;
  RKLLMCallback           callback;
  rknn_gemma4_app_context app_ctx;

  memset(&vocab_info, 0, sizeof(vocab_info));
  memset(&token_embedding, 0, sizeof(token_embedding));
  memset(&input_cb_data, 0, sizeof(input_cb_data));
  memset(&params, 0, sizeof(params));
  memset(&callback, 0, sizeof(callback));
  memset(&app_ctx, 0, sizeof(rknn_gemma4_app_context));
  token_embedding.fd               = -1;
  input_cb_data.per_layer_embed.fd = -1;
  input_cb_data.rope_fd            = -1;

  tokenizer = new Tokenizer(TOKENIZER_BACKEND_LLAMA, tokenizer_path_s.c_str());
  if (!tokenizer) {
    fprintf(stderr, "[server] load tokenizer failed: %s\n", tokenizer_path_s.c_str());
    ret = -1;
    goto srv_out;
  }
  tokenizer->GetVocabInfo(&vocab_info);

  ret = load_embedding(embedding_path_s.c_str(), vocab_info.vocab_size, &token_embedding);
  if (ret != 0) goto srv_out;
  ret = load_embedding(per_layer_s.c_str(), vocab_info.vocab_size, &input_cb_data.per_layer_embed);
  if (ret != 0) goto srv_out;

  params.logits_name                 = (char*)"logits_gathered";
  params.max_context_len             = 0;
  params.sampling_param              = SAMPLE_PARAMS;
  params.vocab_info.vocab_size       = vocab_info.vocab_size;
  params.vocab_info.n_special_eos_id = vocab_info.n_special_eos_id;
  params.vocab_info.n_special_bos_id = vocab_info.n_special_bos_id;
  params.vocab_info.linefeed_id      = vocab_info.linefeed_id;
  params.vocab_info.ignore_eos_token = 0;
  memcpy(params.vocab_info.special_eos_id, vocab_info.special_eos_id, sizeof(vocab_info.special_eos_id));
  memcpy(params.vocab_info.special_bos_id, vocab_info.special_bos_id, sizeof(vocab_info.special_bos_id));

  callback.result_callback    = result_callback;
  callback.result_userdata    = tokenizer;
  callback.embed_callback     = embed_callback;
  callback.embed_userdata     = &token_embedding;
  callback.tokenizer_callback = tokenizer_callback;
  callback.tokenizer_userdata = tokenizer;
  callback.output_callback    = output_callback;
  callback.output_userdata    = &token_embedding;
  callback.input_callback     = input_callback;
  callback.input_userdata     = &input_cb_data;

  ret = init_gemma4_model(&app_ctx, llm_model.c_str(), llm_weight.c_str(),
                          audio_model_s.c_str(), audio_weight_s.c_str(),
                          enable_audio, enable_vision,
                          &params, 1, callback, llm_core_mask, audio_core_mask, 0,
                          per_layer_s.c_str(), NULL, tokenizer, &token_embedding, &input_cb_data,
                          NULL, NULL);
  if (ret != RKNN3_SUCCESS) {
    fprintf(stderr, "[server] init_gemma4_model fail ret=%d\n", ret);
    goto srv_out;
  }

  // rope cache (host storage) — same gating as main().
  if (app_ctx.llm.llm_config.rope_cache_host_storage) {
    ret = load_safetensors(safetensors_s.c_str(), input_cb_data.rope_caches,
                           &input_cb_data.rope_fd, &input_cb_data.rope_mmap_base,
                           &input_cb_data.rope_mmap_size);
    if (ret != 0) {
      fprintf(stderr, "[server] load_safetensors fail: %s\n", safetensors_s.c_str());
      goto srv_out;
    }
  }

  if (max_context_len > app_ctx.llm.llm_config.max_ctx_len) {
    fprintf(stderr, "[server] warn: max_context_len %d > model max_ctx_len %d\n",
            max_context_len, app_ctx.llm.llm_config.max_ctx_len);
  }

  // Re-route stdout: dup the real stdout fd for raw frames, point C stdout at
  // stderr so any stray printf() never corrupts the token byte stream.
  g_frame_fd = dup(STDOUT_FILENO);
  if (g_frame_fd < 0) {
    fprintf(stderr, "[server] dup(stdout) failed\n");
    ret = -1;
    goto srv_out;
  }
  if (dup2(STDERR_FILENO, STDOUT_FILENO) < 0) {
    fprintf(stderr, "[server] dup2 stderr->stdout failed\n");
    ret = -1;
    goto srv_out;
  }
  g_server_mode = true;

  // Handshake: Init complete, advertise protocol version.
  fprintf(stderr, "[server] Init complete\n");
  fprintf(stderr, "READY %d\n", AUDIO_LLM_PROTOCOL_VERSION);
  fflush(stderr);

  {
    std::string line;
    int utt = 0;
    while (std::getline(std::cin, line)) {
      while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
      if (line.empty()) {
        // Empty request line: emit an EOS so the client unblocks, keep serving.
        emit_eos_frame();
        continue;
      }

      std::string audio_ref;
      std::string prompt_str;
      int32_t     req_max_new = 256;
      try {
        nlohmann::json req = nlohmann::json::parse(line);
        if (req.contains("audio_ref") && !req["audio_ref"].is_null())
          audio_ref = req["audio_ref"].get<std::string>();
        if (req.contains("prompt") && !req["prompt"].is_null())
          prompt_str = req["prompt"].get<std::string>();
        if (req.contains("max_new_tokens") && !req["max_new_tokens"].is_null())
          req_max_new = req["max_new_tokens"].get<int32_t>();
      } catch (const nlohmann::json::exception& e) {
        fprintf(stderr, "[server] bad JSON request: %s (%s)\n", line.c_str(), e.what());
        emit_eos_frame();
        continue;
      }

      // If the prompt has no <audio*> marker but audio_ref is set, prepend the
      // standard audio-transcribe scaffold so the encoder output is consumed.
      if (enable_audio && !audio_ref.empty() &&
          prompt_str.find("<audio") == std::string::npos) {
        prompt_str = std::string("<audio>") + prompt_str;
      }

      fprintf(stderr, "[server] req#%d audio_ref=\"%s\" prompt=\"%s\" max_new=%d\n",
              utt, audio_ref.c_str(), prompt_str.c_str(), req_max_new);
      int rc = run_server_request(&app_ctx, enable_audio, audio_ref, prompt_str, req_max_new);
      if (rc != 0) fprintf(stderr, "[server] req#%d FAILED rc=%d\n", utt, rc);
      // Always reset the KV cache between requests, regardless of rc.  The
      // success path inside inference_gemma4_llm already clears KV, but its
      // error early-returns skip it; without this unconditional reset a failed
      // request leaves stale KV that accumulates across turns until the context
      // overflows and the runtime aborts (SIGABRT / rc=-6) on a later request.
      gemma4_server_reset_kvcache(&app_ctx);
      emit_eos_frame();
      fprintf(stderr, "[server] req#%d done\n", utt);
      utt++;
    }
  }

  ret = 0;

srv_out:
  release_gemma4_model(&app_ctx);
  release_safetensors(&input_cb_data);
  release_embedding(&input_cb_data.per_layer_embed);
  release_embedding(&token_embedding);
  if (tokenizer) { delete tokenizer; tokenizer = NULL; }
  return ret;
}

int main(int argc, char** argv)
{
  // ── Server mode dispatch (opt-in): "<model_dir> [--device-id <id>] -" ──
  // The trailing "-" sentinel (matching the TTS server mode + the Python
  // AudioLLMWorker._build_args) selects the persistent AudioLLM server. The
  // default one-shot 18-positional-arg path below is unchanged.
  if (argc >= 2 && std::string(argv[argc - 1]) == "-") {
    std::string model_dir;
    std::string device_id;
    for (int i = 1; i < argc - 1; ++i) {
      std::string a = argv[i];
      if (a == "--device-id" && i + 1 < argc - 1) {
        device_id = argv[++i];
      } else if (model_dir.empty()) {
        model_dir = a;
      }
    }
    if (model_dir.empty()) {
      fprintf(stderr, "server mode usage: %s <model_dir> [--device-id <id>] -\n", argv[0]);
      return -1;
    }
    return run_server(model_dir, device_id);
  }

  if (argc < 18 || argc > 19) {
    print_usage(argv[0]);
    return -1;
  }

  const char* llm_model_path           = argv[1];
  const char* llm_weight_path          = argv[2];
  const uint32_t llm_core_mask         = strtoul(argv[3], NULL, 16);
  const char* tokenizer_path           = argv[4];
  const char* embedding_path           = argv[5];
  int32_t     max_context_len          = atoi(argv[6]);
  int32_t     max_new_tokens           = atoi(argv[7]);
  const char* per_layer_embed_path     = argv[8];
  const char* safetensors_path         = argv[9];
  const char* audio_model_path         = argv[10];
  const char* audio_weight_path        = argv[11];
  const uint32_t audio_core_mask       = strtoul(argv[12], NULL, 16);
  const char* vision_model_path        = argv[13];
  const char* vision_weight_path       = argv[14];
  const uint32_t vision_core_mask      = strtoul(argv[15], NULL, 16);
  const char* audio_path               = argv[16];
  const char* image_path               = argv[17];
  const char* prompt                   = (argc >= 19) ? argv[18] : NULL;

  bool enable_audio = !is_empty_arg(audio_model_path) && !is_empty_arg(audio_weight_path);
  bool enable_vision = !is_empty_arg(vision_model_path) && !is_empty_arg(vision_weight_path);
  bool audio_pair_empty = is_empty_arg(audio_model_path) && is_empty_arg(audio_weight_path);
  bool vision_pair_empty = is_empty_arg(vision_model_path) && is_empty_arg(vision_weight_path);

  if (!enable_audio && !audio_pair_empty) {
    LOGW("Error: audio rknn and audio weight must be both provided or both empty strings.\n");
    print_usage(argv[0]);
    return -1;
  }

  if (!enable_vision && !vision_pair_empty) {
    LOGW("Error: vision rknn and vision weight must be both provided or both empty strings.\n");
    print_usage(argv[0]);
    return -1;
  }

  int                     ret = 0;
  VocabInfo               vocab_info;
  Tokenizer*              tokenizer = NULL;
  embedding_info          token_embedding;
  input_cb_userdata       input_cb_data;
  rknn3_llm_param         params;
  RKLLMCallback           callback;
  rknn_gemma4_app_context app_ctx;
  rknn_perf_metrics_t     perf;
  audio_buffer_t          src_audio;
  image_buffer_t          src_image;
  float16*                audio_embeds   = NULL;
  float16*                image_embeds   = NULL;
  size_t                  n_image_tokens = 0;
  size_t                  n_embed_image  = 0;
  size_t                  n_audio_tokens = 0;
  size_t                  n_embed_audio  = 0;
  int                     n_audio        = 0;
  int                     n_image        = 0;

  memset(&vocab_info, 0, sizeof(vocab_info));
  memset(&token_embedding, 0, sizeof(token_embedding));
  memset(&input_cb_data, 0, sizeof(input_cb_data));
  memset(&params, 0, sizeof(params));
  memset(&src_audio, 0, sizeof(audio_buffer_t));
  memset(&src_image, 0, sizeof(image_buffer_t));
  memset(&callback, 0, sizeof(callback));
  memset(&app_ctx, 0, sizeof(rknn_gemma4_app_context));
  memset(&perf, 0, sizeof(perf));
  token_embedding.fd               = -1;
  input_cb_data.per_layer_embed.fd = -1;
  input_cb_data.rope_fd            = -1;

  // LLM Multi Model Tensor
  rknn3_llm_multimodal_tensor tensor;
  memset(&tensor, 0, sizeof(rknn3_llm_multimodal_tensor));

  tokenizer = new Tokenizer(TOKENIZER_BACKEND_LLAMA, tokenizer_path);
  if (!tokenizer) {
    printf("load tokenizer failed! tokenizer_path=%s\n", tokenizer_path);
    ret = -1;
    goto out;
  }
  tokenizer->GetVocabInfo(&vocab_info);
  print_vocab_info(&vocab_info);

  ret = load_embedding(embedding_path, vocab_info.vocab_size, &token_embedding);
  if (ret != 0) {
    goto out;
  }

  ret = load_embedding(per_layer_embed_path, vocab_info.vocab_size, &input_cb_data.per_layer_embed);
  if (ret != 0) {
    goto out;
  }

  params.logits_name                      = (char*)"logits_gathered";
  params.max_context_len                  = 0;
  params.sampling_param                   = SAMPLE_PARAMS;
  params.vocab_info.vocab_size            = vocab_info.vocab_size;
  params.vocab_info.n_special_eos_id      = vocab_info.n_special_eos_id;
  params.vocab_info.n_special_bos_id      = vocab_info.n_special_bos_id;
  params.vocab_info.linefeed_id           = vocab_info.linefeed_id;
  params.vocab_info.ignore_eos_token      = 0;
  memcpy(params.vocab_info.special_eos_id, vocab_info.special_eos_id, sizeof(vocab_info.special_eos_id));
  memcpy(params.vocab_info.special_bos_id, vocab_info.special_bos_id, sizeof(vocab_info.special_bos_id));

  callback.result_callback    = result_callback;
  callback.result_userdata    = tokenizer;
  callback.embed_callback     = embed_callback;
  callback.embed_userdata     = &token_embedding;
  callback.tokenizer_callback = tokenizer_callback;
  callback.tokenizer_userdata = tokenizer;
  callback.output_callback    = output_callback;
  callback.output_userdata    = &token_embedding;
  callback.input_callback     = input_callback;
  callback.input_userdata     = &input_cb_data;

  ret = init_gemma4_model(&app_ctx, llm_model_path, llm_weight_path, audio_model_path, audio_weight_path,
                          enable_audio, enable_vision,
                          &params, 1, callback, llm_core_mask, audio_core_mask, vision_core_mask,
                          per_layer_embed_path, NULL, tokenizer, &token_embedding, &input_cb_data,
                          vision_model_path, vision_weight_path);
  if (ret != RKNN3_SUCCESS) {
    printf("init_gemma4_model fail! ret=%d llm_model_path=%s llm_weight_path=%s\n", ret, llm_model_path, llm_weight_path);
    goto out;
  }

  // Read audio/image files if provided.
  if (enable_audio &&
      audio_path != NULL && strlen(audio_path) > 0) {
    printf("--> reading audio file: %s\n", audio_path);
    ret = read_audio(audio_path, &src_audio);
    if (ret != 0) {
      printf("read_audio fail! ret=%d audio_path=%s\n", ret, audio_path);
      goto out;
    }
    printf("audio: num_frames=%d, num_channels=%d, sample_rate=%d\n",
           src_audio.num_frames, src_audio.num_channels, src_audio.sample_rate);

    // Long-audio chunking: size the embed buffer for N_chunks worth of audio
    // tokens. Each <=7 s chunk yields at most the largest output bucket worth of
    // tokens; size every chunk slot at that max bucket (a safe upper bound).
    {
      int n_chunks = gemma4_audio_num_chunks(src_audio.num_frames);
      if (n_chunks < 1) n_chunks = 1;
      // Per-chunk upper bound = the largest output bucket across all shapes
      // (189 tokens for the 7.5 s bucket). Robust regardless of shape ordering.
      int per_chunk_tokens = 0;
      for (int s = 0; s < app_ctx.audio.n_shapes; s++) {
        if (app_ctx.audio.embeds_dim0[s] > per_chunk_tokens) {
          per_chunk_tokens = app_ctx.audio.embeds_dim0[s];
        }
      }
      if (per_chunk_tokens <= 0) {
        per_chunk_tokens = get_n_audio(&app_ctx.audio, src_audio.num_frames);
      }
      n_audio_tokens = (size_t)n_chunks * (size_t)per_chunk_tokens;
      printf("audio buffer sizing: num_frames=%d -> %d chunk(s) x %d tokens = %zu max audio tokens\n",
             src_audio.num_frames, n_chunks, per_chunk_tokens, n_audio_tokens);
    }
    n_embed_audio = n_audio_tokens * app_ctx.audio.embeds_dim1;
    audio_embeds = (float16*)malloc(n_embed_audio * sizeof(float16));
    if (!audio_embeds) {
      printf("Failed to allocate audio_embeds\n");
      ret = -1;
      goto out;
    }
  }

  if (enable_vision &&
      image_path != NULL && strlen(image_path) > 0) {
    printf("--> reading image file: %s\n", image_path);
    ret = read_image(image_path, &src_image);
    if (ret != 0) {
      printf("read_image fail! ret=%d image_path=%s\n", ret, image_path);
      goto out;
    }

    n_image_tokens = get_vision_n_tokens_main(&app_ctx.vision);
    n_embed_image = app_ctx.vision.outputs[0].mem->size / sizeof(float16);
    image_embeds = (float16*)malloc(n_embed_image * sizeof(float16));
    if (!image_embeds) {
      printf("Failed to allocate image_embeds\n");
      ret = -1;
      goto out;
    }
  }

  // Determine safetensors_path and prompt based on rope_cache_host_storage.
  if (app_ctx.llm.llm_config.rope_cache_host_storage) {
    // rope cache required
    if (safetensors_path == NULL || strlen(safetensors_path) <= 0) {
      LOGW("Error: model requires rope_caches.safetensors (rope_cache_host_storage=1), "
           "but <safetensors_path> not provided\n");
      print_usage(argv[0]);
      ret = -1;
      goto out;
    }
    ret = load_safetensors(safetensors_path, input_cb_data.rope_caches,
                           &input_cb_data.rope_fd, &input_cb_data.rope_mmap_base,
                           &input_cb_data.rope_mmap_size);
    if (ret != 0) {
      goto out;
    }
  } else {
    // rope cache not required
    LOGW("Warning: extra arguments ignored (rope_cache_host_storage=0, safetensors not needed)\n");
  }

  if (prompt == NULL || strlen(prompt) <= 0) {
    LOGW("Warning: no prompt provided\n");
    goto out;
  }

  if (max_context_len != app_ctx.llm.llm_config.max_ctx_len) {
    if (max_context_len < app_ctx.llm.llm_config.max_ctx_len) {
      LOGW("Warning: max_context_len (%d) is less than llm_config.max_ctx_len (%d).\n", max_context_len,
           app_ctx.llm.llm_config.max_ctx_len);
      LOGW("It's recommended to set <max_context_len> to %d.\n", app_ctx.llm.llm_config.max_ctx_len);
    } else {
      LOGW("Error: max_context_len (%d) is greater than llm_config.max_ctx_len (%d).\n", max_context_len,
           app_ctx.llm.llm_config.max_ctx_len);
      LOGW("Please set <max_context_len> to %d.\n", app_ctx.llm.llm_config.max_ctx_len);
      ret = -1;
      goto out;
    }
  }

  print_llm_config(&app_ctx.llm.llm_config, max_new_tokens);

  if (enable_audio &&
      (strstr(prompt, "<audio>") || strstr(prompt, "<|audio>") ||
       strstr(prompt, "<|audio|>") || strstr(prompt, "<audio|>"))) {
    n_audio = 1;
  }
  if (enable_vision &&
      (strstr(prompt, "<image>") || strstr(prompt, "<|image>") ||
       strstr(prompt, "<|image|>") || strstr(prompt, "<image|>") ||
       strstr(prompt, "<start_of_image>"))) {
    n_image = 1;
  }

  // LLM Input
  tensor.name                 = "input_embeds";
  tensor.prompt               = prompt;
  tensor.enable_thinking = false;
  if (enable_audio) {
    tensor.audio.audio_embed    = audio_embeds;
    tensor.audio.n_audio_tokens = 0; // update after audio encoder inference
    tensor.audio.n_audio        = n_audio;
    tensor.audio.audio_start    = "<|audio>";
    tensor.audio.audio_end      = "<audio|>";
    tensor.audio.audio_content  = "<|audio|>";
  }

  if (enable_vision) {
    tensor.image.image_embed    = image_embeds;
    tensor.image.n_image_tokens = n_image_tokens;
    tensor.image.n_image        = n_image;
    tensor.image.image_width    = app_ctx.vision.model_width;
    tensor.image.image_height   = app_ctx.vision.model_height;
    tensor.image.image_start    = "<|image>";
    tensor.image.image_end      = "<image|>";
    tensor.image.image_content  = "<|image|>";
  }

  printf("--> inference gemma4 model\n");
  g_first_decode        = true;
  g_first_token_time_us = 0;
  ret = inference_gemma4_model(&app_ctx, tensor, &src_audio, &src_image,
                             audio_embeds, image_embeds, max_new_tokens, &perf);
  if (ret != RKNN3_SUCCESS) {
    printf("inference_gemma4_model fail! ret=%d\n", ret);
    goto out;
  }

  print_perf(&perf);

out:
  if (audio_embeds) {
    free(audio_embeds);
    audio_embeds = NULL;
  }
  if (image_embeds) {
    free(image_embeds);
    image_embeds = NULL;
  }
  if (src_audio.data) {
    free(src_audio.data);
    src_audio.data = NULL;
  }
  if (src_image.virt_addr) {
    free(src_image.virt_addr);
    src_image.virt_addr = NULL;
  }
  release_gemma4_model(&app_ctx);
  release_safetensors(&input_cb_data);
  release_embedding(&input_cb_data.per_layer_embed);
  release_embedding(&token_embedding);

  if (tokenizer) {
    delete tokenizer;
    tokenizer = NULL;
  }

  return ret;
}
