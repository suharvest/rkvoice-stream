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
#include "common.h"
#include "image_utils.h"
#include "audio_utils.h"
#include "nlohmann/json.hpp"

#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <time.h>

// Long-audio chunking: the Gemma-4 audio encoder is capped at 756 mel frames
// (~7.5 s) / 189 audio tokens per single call. For audio longer than that we
// split the waveform into <=GEMMA4_MAX_CHUNK_SAMPLES windows, encode each with
// inference_gemma4_audio, and concatenate the *valid* (non-padding) embeds in
// order. 112000 samples = 7.0 s @ 16 kHz leaves margin under the 7.5 s bucket.
#define GEMMA4_MAX_CHUNK_SAMPLES 112000

int gemma4_audio_num_chunks(int num_frames)
{
    if (num_frames <= 0) return 0;
    return (num_frames + GEMMA4_MAX_CHUNK_SAMPLES - 1) / GEMMA4_MAX_CHUNK_SAMPLES;
}

// -----------------------------------------------------------------------------
// Streaming-pipeline demo helper: encode ONE audio chunk [chunk_idx] of the full
// waveform and append its valid (front-loaded) embed rows into audio_embeds at
// row offset *total_valid_io, advancing *total_valid_io. This is the exact same
// per-chunk encode the batched long-audio path does inside inference_gemma4_model,
// factored out so the GEMMA4_STREAM_DEMO path can encode chunks one at a time
// (some "during speech", the last "post speech") while concatenating into the
// SAME contiguous embeds buffer -> a single, layout-faithful prefill afterwards.
// Returns the encode wall time in microseconds (>=0) or <0 on error.
static int64_t gemma4_encode_one_chunk(rknn_gemma4_app_context* app_ctx,
                                       audio_buffer_t* audio, int chunk_idx,
                                       int n_chunks, float16* chunk_scratch,
                                       float16* audio_embeds, int* total_valid_io)
{
    const int embeds_dim1 = app_ctx->audio.embeds_dim1;
    int offset = chunk_idx * GEMMA4_MAX_CHUNK_SAMPLES;
    int len    = audio->num_frames - offset;
    if (len > GEMMA4_MAX_CHUNK_SAMPLES) len = GEMMA4_MAX_CHUNK_SAMPLES;

    audio_buffer_t chunk;
    chunk.data         = audio->data + offset;
    chunk.num_frames   = len;
    chunk.num_channels = audio->num_channels;
    chunk.sample_rate  = audio->sample_rate;

    int     chunk_valid = 0;
    int64_t t0  = getCurrentTimeUs();
    int     ret = inference_gemma4_audio(&app_ctx->audio, &chunk, chunk_scratch, &chunk_valid);
    int64_t t1  = getCurrentTimeUs();
    if (ret != 0) {
        printf("[STREAM-DEMO] encode chunk %d/%d failed ret=%d\n", chunk_idx + 1, n_chunks, ret);
        return -1;
    }
    memcpy(audio_embeds + (size_t)(*total_valid_io) * (size_t)embeds_dim1,
           chunk_scratch,
           (size_t)chunk_valid * (size_t)embeds_dim1 * sizeof(float16));
    *total_valid_io += chunk_valid;
    int64_t us = t1 - t0;
    printf("[STREAM-DEMO]   chunk %d/%d: samples [%d,%d) -> %d tokens, encode %.1f ms\n",
           chunk_idx + 1, n_chunks, offset, offset + len, chunk_valid, (double)us / 1000.0);
    return us;
}

// Opt-in streaming-pipeline demo (env GEMMA4_STREAM_DEMO=1). Single process,
// simulates chunked arrival of a long utterance to prove the post-speech latency
// win of overlapping audio-encode with "speech in progress". Runs the SAME audio
// + SAME question twice:
//   MODE A (batch / status quo): nothing happens until the speaker finishes;
//     then ALL chunks are encoded + a single prefill + generate. The whole audio
//     encode is on the post-speech critical path.
//   MODE B (pipeline): chunks 1..N-1 are encoded WHILE the user is still talking
//     (off the post-speech path); when speech ends only the LAST chunk is encoded,
//     then the single (identical) prefill + generate.
// The prefill is ONE session_run over the full <audio...>question turn in BOTH
// modes (layout-faithful, identical KV), so generate output is byte-identical;
// only the encode work on the post-speech path differs.
static int gemma4_stream_demo(rknn_gemma4_app_context* app_ctx,
                              rknn3_llm_multimodal_tensor tensor,
                              audio_buffer_t* audio, float16* audio_embeds,
                              int32_t max_new_tokens, rknn_perf_metrics_t* perf)
{
    const int embeds_dim1 = app_ctx->audio.embeds_dim1;
    int n_chunks = gemma4_audio_num_chunks(audio ? audio->num_frames : 0);
    if (n_chunks < 1) {
        printf("[STREAM-DEMO] no audio to stream\n");
        return -1;
    }

    int max_dim0 = 0;
    for (int s = 0; s < app_ctx->audio.n_shapes; s++) {
        if (app_ctx->audio.embeds_dim0[s] > max_dim0) max_dim0 = app_ctx->audio.embeds_dim0[s];
    }
    float16* scratch = (float16*)malloc((size_t)max_dim0 * (size_t)embeds_dim1 * sizeof(float16));
    if (!scratch) { printf("[STREAM-DEMO] scratch alloc failed\n"); return -1; }

    printf("\n[STREAM-DEMO] %d samples (%.3fs), %d chunk(s). max_new_tokens=%d\n",
           audio->num_frames, (float)audio->num_frames / 16000.0f, n_chunks, max_new_tokens);

    // ======================= MODE A: BATCH (status quo) ======================
    // Post-speech timer covers: encode ALL chunks + the single prefill+generate.
    printf("\n[STREAM-DEMO] ===== MODE A (batch): all encode is POST-speech =====\n");
    int     a_valid = 0;
    int64_t a_post0 = getCurrentTimeUs();        // speech just ended
    int64_t a_enc_us = 0;
    for (int c = 0; c < n_chunks; c++) {
        int64_t us = gemma4_encode_one_chunk(app_ctx, audio, c, n_chunks, scratch, audio_embeds, &a_valid);
        if (us < 0) { free(scratch); return -1; }
        a_enc_us += us;
    }
    rknn3_llm_multimodal_tensor tA = tensor;
    tA.audio.audio_embed    = audio_embeds;
    tA.audio.n_audio_tokens = a_valid;
    tA.audio.n_audio        = 1;
    rknn_perf_metrics_t perfA; memset(&perfA, 0, sizeof(perfA));
    int ret = inference_gemma4_llm(&app_ctx->llm, tA, max_new_tokens, &perfA);
    if (ret != 0) { free(scratch); return ret; }
    int64_t a_post1 = getCurrentTimeUs();
    rknn3_session_clear_kvcache(app_ctx->llm.rknn_sess, RKNN3_KVCACHE_CLEAR_ALL);
    double a_post_ms   = (double)(a_post1 - a_post0) / 1000.0;
    double a_enc_ms    = (double)a_enc_us / 1000.0;
    double a_llm_ms    = (double)(perfA.llm_end_time - perfA.llm_start_time) / 1000.0;

    // ======================= MODE B: PIPELINE ================================
    // Chunks 1..N-1 are encoded DURING speech (off the post-speech path). Only
    // the last chunk's encode + the single prefill+generate are post-speech.
    printf("\n[STREAM-DEMO] ===== MODE B (pipeline): chunks 1..%d encode DURING speech =====\n",
           n_chunks - 1);
    int     b_valid = 0;
    int64_t b_during_us = 0;
    for (int c = 0; c < n_chunks - 1; c++) {
        printf("[STREAM-DEMO] [during speech] ");
        int64_t us = gemma4_encode_one_chunk(app_ctx, audio, c, n_chunks, scratch, audio_embeds, &b_valid);
        if (us < 0) { free(scratch); return -1; }
        b_during_us += us;
    }
    // ---- speech ends here: start the post-speech timer ----
    int64_t b_post0 = getCurrentTimeUs();
    int64_t b_last_enc_us = 0;
    if (n_chunks >= 1) {
        printf("[STREAM-DEMO] [post speech] ");
        int64_t us = gemma4_encode_one_chunk(app_ctx, audio, n_chunks - 1, n_chunks, scratch, audio_embeds, &b_valid);
        if (us < 0) { free(scratch); return -1; }
        b_last_enc_us = us;
    }
    rknn3_llm_multimodal_tensor tB = tensor;
    tB.audio.audio_embed    = audio_embeds;
    tB.audio.n_audio_tokens = b_valid;
    tB.audio.n_audio        = 1;
    rknn_perf_metrics_t perfB; memset(&perfB, 0, sizeof(perfB));
    ret = inference_gemma4_llm(&app_ctx->llm, tB, max_new_tokens, &perfB);
    if (ret != 0) { free(scratch); return ret; }
    int64_t b_post1 = getCurrentTimeUs();
    rknn3_session_clear_kvcache(app_ctx->llm.rknn_sess, RKNN3_KVCACHE_CLEAR_ALL);
    double b_post_ms     = (double)(b_post1 - b_post0) / 1000.0;
    double b_during_ms   = (double)b_during_us / 1000.0;
    double b_last_enc_ms = (double)b_last_enc_us / 1000.0;
    double b_llm_ms      = (double)(perfB.llm_end_time - perfB.llm_start_time) / 1000.0;

    free(scratch);

    printf("\n========================  STREAM-DEMO POST-SPEECH LATENCY  ========================\n");
    printf(" Audio: %.3fs split into %d chunk(s). Prefill is ONE session_run in both modes.\n",
           (float)audio->num_frames / 16000.0f, n_chunks);
    printf(" %-40s | %-12s | %-12s\n", "", "MODE A batch", "MODE B pipe");
    printf("----------------------------------------------------------------------------------\n");
    printf(" %-40s | %-12d | %-12d\n", "audio tokens (concatenated)", a_valid, b_valid);
    printf(" %-40s | %-12llu | %-12llu\n", "prefill tokens (audio+text)",
           (unsigned long long)perfA.n_prefill_tokens, (unsigned long long)perfB.n_prefill_tokens);
    printf(" %-40s | %-12.1f | %-12.1f\n", "encode DURING speech (off path) ms", 0.0, b_during_ms);
    printf(" %-40s | %-12.1f | %-12.1f\n", "encode POST speech (on path) ms", a_enc_ms, b_last_enc_ms);
    printf(" %-40s | %-12.1f | %-12.1f\n", "LLM prefill+generate ms", a_llm_ms, b_llm_ms);
    printf("----------------------------------------------------------------------------------\n");
    printf(" %-40s | %-12.1f | %-12.1f\n", "POST-SPEECH TOTAL ms (lower=better)", a_post_ms, b_post_ms);
    printf("----------------------------------------------------------------------------------\n");
    if (a_post_ms > 0.0) {
        printf(" Pipeline saved %.1f ms post-speech (%.1f%%) by overlapping %d/%d chunk encodes with speech.\n",
               a_post_ms - b_post_ms, 100.0 * (a_post_ms - b_post_ms) / a_post_ms, n_chunks - 1, n_chunks);
    }
    printf("==================================================================================\n\n");

    // Propagate MODE B (the streaming path) metrics out.
    perf->n_prefill_tokens = perfB.n_prefill_tokens;
    perf->n_decode_tokens  = perfB.n_decode_tokens;
    perf->llm_start_time   = perfB.llm_start_time;
    perf->llm_end_time     = perfB.llm_end_time;
    perf->audio_latency    = b_last_enc_us;   // post-speech encode only
    return 0;
}

// =============================================================================
// REAL incremental streaming loop (env GEMMA4_RTSTREAM=1).
//
// Unlike gemma4_stream_demo (GEMMA4_STREAM_DEMO) which already has the whole
// waveform resident and merely *slices* it off-path, this path drives a genuine
// producer/consumer loop: a "reader" feeds samples into an accumulation buffer
// in small steps (simulating real-time arrival; with GEMMA4_RT_PACE=1 it sleeps
// so the stream is released at true wall-clock speed, e.g. 7 s of audio only
// becomes available after ~7 s). The instant the accumulator reaches one 7 s
// chunk, that chunk is encoded IMMEDIATELY into the shared audio_embeds buffer
// -- so chunks 1..N-1 finish encoding *while the next chunk is still being
// captured* (encode ~172 ms << 7 s capture, naturally hidden). Only when the
// stream ends (EOF / tail chunk) do we encode the final chunk and run the SINGLE
// layout-faithful inference_gemma4_llm prefill+generate.
//
// The file reader here is the stand-in source; swapping it for a live mic ring
// buffer (same "append samples, encode on fill" contract) is a drop-in change.
//
// Shared with main.cc: g_first_token_time_us is set by the LLM decode callback
// at the first response token. We publish the stream-end timestamp here so main
// can report the true post-stream latency = (first response token) - (stream end).
int64_t g_gemma4_stream_end_us = 0;  // wall-clock us when the audio stream ended
// Set by the LLM decode callback in main.cc at the first response token.
extern int64_t g_first_token_time_us;

// Sleep helper (microseconds) for RT_PACE; nanosleep avoids usleep's signal
// quirks and gives ms-grade pacing fidelity.
static void gemma4_sleep_us(int64_t us)
{
    if (us <= 0) return;
    struct timespec ts;
    ts.tv_sec  = us / 1000000;
    ts.tv_nsec = (us % 1000000) * 1000;
    nanosleep(&ts, NULL);
}

static int gemma4_rtstream_loop(rknn_gemma4_app_context* app_ctx,
                                rknn3_llm_multimodal_tensor tensor,
                                audio_buffer_t* audio, float16* audio_embeds,
                                int32_t max_new_tokens, rknn_perf_metrics_t* perf)
{
    const int embeds_dim1 = app_ctx->audio.embeds_dim1;
    const int total_samples = audio ? audio->num_frames : 0;
    int n_chunks = gemma4_audio_num_chunks(total_samples);
    if (n_chunks < 1) { printf("[RTSTREAM] no audio to stream\n"); return -1; }

    const bool pace = []{ const char* e = getenv("GEMMA4_RT_PACE"); return e && e[0] == '1'; }();

    // Reader granularity: deliver the waveform in 0.5 s steps so the loop polls
    // the accumulator frequently (mirrors a mic delivering 10-30 ms frames; 0.5 s
    // keeps the log readable). With pace=on, each step sleeps step_dur in wall.
    const int   READ_STEP_SAMPLES = 8000;            // 0.5 s @ 16 kHz
    const double STEP_DUR_S        = (double)READ_STEP_SAMPLES / 16000.0;

    // Scratch for ONE chunk encode (largest output bucket across shapes).
    int max_dim0 = 0;
    for (int s = 0; s < app_ctx->audio.n_shapes; s++)
        if (app_ctx->audio.embeds_dim0[s] > max_dim0) max_dim0 = app_ctx->audio.embeds_dim0[s];
    float16* scratch = (float16*)malloc((size_t)max_dim0 * (size_t)embeds_dim1 * sizeof(float16));
    if (!scratch) { printf("[RTSTREAM] scratch alloc failed\n"); return -1; }

    printf("\n[RTSTREAM] real incremental loop: %d samples (%.3fs), %d chunk(s), pace=%s\n",
           total_samples, (float)total_samples / 16000.0f, n_chunks, pace ? "ON" : "off");

    const int64_t t_stream_start = getCurrentTimeUs();
    int total_valid   = 0;   // audio tokens emitted so far (rows in audio_embeds)
    int chunks_done   = 0;   // chunks fully encoded
    int delivered     = 0;   // samples handed to the accumulator by the reader
    int chunk_base    = 0;   // sample offset of the in-progress chunk

    // ---- producer/consumer loop: read a step, then encode any chunk now full ----
    while (delivered < total_samples) {
        int step = READ_STEP_SAMPLES;
        if (delivered + step > total_samples) step = total_samples - delivered;
        // Simulate this step "arriving" over the wire / off the mic.
        if (pace) gemma4_sleep_us((int64_t)(STEP_DUR_S * 1e6));
        delivered += step;

        // Encode every chunk whose 7 s window is now fully captured. (A single
        // step can complete at most one chunk, but loop to be safe.)
        while (delivered - chunk_base >= GEMMA4_MAX_CHUNK_SAMPLES &&
               chunks_done < n_chunks - 1) {
            int len = GEMMA4_MAX_CHUNK_SAMPLES;
            audio_buffer_t chunk;
            chunk.data         = audio->data + chunk_base;
            chunk.num_frames   = len;
            chunk.num_channels = audio->num_channels;
            chunk.sample_rate  = audio->sample_rate;

            int chunk_valid = 0;
            int64_t e0 = getCurrentTimeUs();
            int ret = inference_gemma4_audio(&app_ctx->audio, &chunk, scratch, &chunk_valid);
            int64_t e1 = getCurrentTimeUs();
            if (ret != 0) { printf("[RTSTREAM] encode chunk %d failed ret=%d\n", chunks_done + 1, ret);
                            free(scratch); return -1; }
            memcpy(audio_embeds + (size_t)total_valid * (size_t)embeds_dim1, scratch,
                   (size_t)chunk_valid * (size_t)embeds_dim1 * sizeof(float16));
            total_valid += chunk_valid;
            chunk_base  += len;
            chunks_done++;
            double rel_ms = (double)(e1 - t_stream_start) / 1000.0;
            printf("[RTSTREAM] [DURING capture] chunk %d/%d encoded @ t=+%.0fms "
                   "(captured %.1fs of %.1fs so far) -> %d tokens, encode %.1fms\n",
                   chunks_done, n_chunks, rel_ms,
                   (double)delivered / 16000.0, (double)total_samples / 16000.0,
                   chunk_valid, (double)(e1 - e0) / 1000.0);
        }
    }

    // ---- stream END: only the tail chunk's encode + the prefill are on-path ----
    g_gemma4_stream_end_us = getCurrentTimeUs();
    double stream_dur_ms = (double)(g_gemma4_stream_end_us - t_stream_start) / 1000.0;
    printf("[RTSTREAM] *** stream ended @ t=+%.0fms (delivered all %.1fs). "
           "%d/%d chunks already encoded DURING capture. ***\n",
           stream_dur_ms, (double)total_samples / 16000.0, chunks_done, n_chunks);

    // Encode the tail chunk (remaining samples) -- this is the only encode that
    // sits on the post-stream critical path.
    int64_t tail_enc_us = 0;
    if (chunk_base < total_samples) {
        int len = total_samples - chunk_base;
        audio_buffer_t chunk;
        chunk.data         = audio->data + chunk_base;
        chunk.num_frames   = len;
        chunk.num_channels = audio->num_channels;
        chunk.sample_rate  = audio->sample_rate;
        int chunk_valid = 0;
        int64_t e0 = getCurrentTimeUs();
        int ret = inference_gemma4_audio(&app_ctx->audio, &chunk, scratch, &chunk_valid);
        int64_t e1 = getCurrentTimeUs();
        if (ret != 0) { printf("[RTSTREAM] tail encode failed ret=%d\n", ret); free(scratch); return -1; }
        memcpy(audio_embeds + (size_t)total_valid * (size_t)embeds_dim1, scratch,
               (size_t)chunk_valid * (size_t)embeds_dim1 * sizeof(float16));
        total_valid += chunk_valid;
        chunks_done++;
        tail_enc_us = e1 - e0;
        printf("[RTSTREAM] [POST stream] tail chunk %d/%d: %d samples -> %d tokens, encode %.1fms\n",
               chunks_done, n_chunks, len, chunk_valid, (double)tail_enc_us / 1000.0);
    }
    free(scratch);

    // ---- single layout-faithful prefill + generate over ALL audio tokens ----
    rknn3_llm_multimodal_tensor t = tensor;
    t.audio.audio_embed    = audio_embeds;
    t.audio.n_audio_tokens = total_valid;
    t.audio.n_audio        = 1;
    int ret = inference_gemma4_llm(&app_ctx->llm, t, max_new_tokens, perf);
    if (ret != 0) return ret;

    int64_t t_end = getCurrentTimeUs();
    double post_total_ms = (double)(t_end - g_gemma4_stream_end_us) / 1000.0;
    double total_wall_ms = (double)(t_end - t_stream_start) / 1000.0;
    double prefill_ms    = (perf->llm_start_time && g_first_token_time_us)
                           ? (double)(g_first_token_time_us - perf->llm_start_time) / 1000.0 : 0.0;
    // TRUE post-stream latency to first response token (set by main's callback).
    double post_to_first_ms = (g_first_token_time_us && g_gemma4_stream_end_us)
                              ? (double)(g_first_token_time_us - g_gemma4_stream_end_us) / 1000.0 : 0.0;

    printf("\n========================  RTSTREAM TIMING  ========================\n");
    printf(" audio %.3fs, %d chunk(s), %d audio tokens, pace=%s\n",
           (float)total_samples / 16000.0f, n_chunks, total_valid, pace ? "ON" : "off");
    printf(" chunks encoded DURING capture (off post-stream path): %d/%d\n", n_chunks - 1, n_chunks);
    printf(" tail (post-stream) encode ........... %.1f ms\n", (double)tail_enc_us / 1000.0);
    printf(" LLM prefill (to first token) ........ %.1f ms\n", prefill_ms);
    printf(" --------------------------------------------------\n");
    printf(" POST-STREAM latency to first token .. %.1f ms   <-- stream-end -> first reply token\n",
           post_to_first_ms);
    printf(" POST-STREAM total (incl. generate) .. %.1f ms\n", post_total_ms);
    printf(" total wall (stream-start -> done) ... %.1f ms\n", total_wall_ms);
    printf(" --------------------------------------------------\n");
    // Batch contrast: if all N chunks were encoded only AFTER stream-end (status
    // quo), the post-stream encode cost would be ~N x per-chunk instead of 1x.
    if (tail_enc_us > 0) {
        double batch_post_encode_ms = (double)tail_enc_us / 1000.0 * (double)n_chunks; // ~ all chunks
        printf(" BATCH baseline (encode ALL %d post-stream): ~%.1f ms encode on path\n",
               n_chunks, batch_post_encode_ms);
        printf(" RTSTREAM hides %d/%d chunk encodes -> ~%.1f ms removed from post-stream path\n",
               n_chunks - 1, n_chunks, batch_post_encode_ms - (double)tail_enc_us / 1000.0);
    }
    printf("==================================================================\n\n");

    perf->audio_latency = tail_enc_us;  // post-stream encode only
    return 0;
}

int init_gemma4_model(rknn_gemma4_app_context* app_ctx,
                      const char* llm_model_path, const char* llm_weight_path,
                      const char* audio_model_path, const char* audio_weight_path,
                      bool enable_audio, bool enable_vision,
                      rknn3_llm_param* params, int n_params,
                      RKLLMCallback callback, uint32_t llm_core_mask, 
                      uint32_t audio_core_mask, uint32_t vision_core_mask,
                      const char* per_layer_embed_path, const char* safetensors_path,
                      Tokenizer* tokenizer, embedding_info* token_embedding,
                      input_cb_userdata* input_cb_data,
                      const char* vision_model_path, const char* vision_weight_path)
{
    (void)per_layer_embed_path;
    (void)safetensors_path;
    (void)tokenizer;
    (void)token_embedding;
    (void)input_cb_data;

    int ret = 0;
    app_ctx->enable_audio = enable_audio;
    app_ctx->enable_vision = enable_vision;

    if (enable_audio) {
        printf("--> init gemma4 audio model\n");
        ret = init_gemma4_audio(&app_ctx->audio, audio_model_path, audio_weight_path, audio_core_mask);
        if (ret < 0) {
            printf("init_gemma4_audio failed! ret=%d\n", ret);
            return ret;
        }
    }

    if (enable_vision) {
        printf("--> init gemma4 vision model\n");
        ret = init_gemma4_vision(&app_ctx->vision, vision_model_path, vision_weight_path, vision_core_mask);
        if (ret < 0) {
            printf("init_gemma4_vision failed! ret=%d\n", ret);
            if (enable_audio) {
                release_gemma4_audio(&app_ctx->audio);
            }
            return ret;
        }
    }

    printf("--> init gemma4 llm model\n");
    ret = init_gemma4_llm(&app_ctx->llm, llm_model_path, llm_weight_path, params, n_params, &callback, llm_core_mask);
    if (ret < 0) {
        printf("init_gemma4_llm failed! ret=%d\n", ret);
        if (enable_audio) {
            release_gemma4_audio(&app_ctx->audio);
        }
        if (enable_vision) {
            release_gemma4_vision(&app_ctx->vision);
        }
        return ret;
    }

    return ret;
}


int release_gemma4_model(rknn_gemma4_app_context* app_ctx)
{
    release_gemma4_llm(&app_ctx->llm);
    if (app_ctx->enable_audio) {
        release_gemma4_audio(&app_ctx->audio);
    }
    if (app_ctx->enable_vision) {
        release_gemma4_vision(&app_ctx->vision);
    }
    return 0;
}


int inference_gemma4_model(rknn_gemma4_app_context* app_ctx,
                           rknn3_llm_multimodal_tensor tensor,
                           audio_buffer_t* audio,
                           image_buffer_t* image,
                           float16* audio_embeds,
                           float16* image_embeds,
                           int32_t max_new_tokens,
                           rknn_perf_metrics_t* perf)
{
    int ret = 0;

    // Opt-in streaming-pipeline demo. Encodes the long audio chunk-by-chunk with
    // chunks 1..N-1 overlapped "during speech", proving the post-speech latency
    // win vs. the batch path. Default (env unset) behaviour is fully unchanged.
    {
        const char* stream_env = getenv("GEMMA4_STREAM_DEMO");
        if (stream_env && stream_env[0] == '1' &&
            app_ctx->enable_audio && tensor.audio.n_audio && audio && audio->num_frames > 0) {
            if (audio_embeds == NULL) audio_embeds = (float16*)tensor.audio.audio_embed;
            if (audio_embeds == NULL) { printf("audio_embeds is NULL\n"); return -1; }
            return gemma4_stream_demo(app_ctx, tensor, audio, audio_embeds, max_new_tokens, perf);
        }
    }

    // Opt-in REAL incremental streaming loop (env GEMMA4_RTSTREAM=1): drives a
    // genuine reader->accumulator->encode-on-fill pipeline, encoding chunks
    // 1..N-1 during capture and only the tail chunk + prefill post-stream.
    {
        const char* rt_env = getenv("GEMMA4_RTSTREAM");
        if (rt_env && rt_env[0] == '1' &&
            app_ctx->enable_audio && tensor.audio.n_audio && audio && audio->num_frames > 0) {
            if (audio_embeds == NULL) audio_embeds = (float16*)tensor.audio.audio_embed;
            if (audio_embeds == NULL) { printf("audio_embeds is NULL\n"); return -1; }
            return gemma4_rtstream_loop(app_ctx, tensor, audio, audio_embeds, max_new_tokens, perf);
        }
    }

    if (app_ctx->enable_audio && tensor.audio.n_audio) {
        int n_valid_audio = 0;
        int64_t start_us = getCurrentTimeUs();
        if (audio_embeds == NULL) {
            audio_embeds = (float16*)tensor.audio.audio_embed;
        }
        if (audio_embeds == NULL) {
            printf("audio_embeds is NULL\n");
            return -1;
        }
        tensor.audio.audio_embed = audio_embeds;

        int n_chunks = gemma4_audio_num_chunks(audio ? audio->num_frames : 0);
        if (n_chunks <= 1) {
            // Original single-buffer path (<=7.5 s). Behaviour unchanged: encode
            // directly into audio_embeds, n_valid_audio = valid token count.
            ret = inference_gemma4_audio(&app_ctx->audio, audio, audio_embeds, &n_valid_audio);
            if (ret != 0) {
                printf("inference_gemma4_audio failed! ret=%d\n", ret);
                return ret;
            }
        } else {
            // Long-audio chunked path. Encode each <=7 s window into a temp
            // single-bucket buffer, then concatenate the valid (front-loaded)
            // embed rows in order into audio_embeds. n_valid_audio accumulates.
            const int embeds_dim1 = app_ctx->audio.embeds_dim1;
            // Largest output bucket across shapes -> temp buffer size.
            int max_dim0 = 0;
            for (int s = 0; s < app_ctx->audio.n_shapes; s++) {
                if (app_ctx->audio.embeds_dim0[s] > max_dim0) {
                    max_dim0 = app_ctx->audio.embeds_dim0[s];
                }
            }
            if (max_dim0 <= 0 || embeds_dim1 <= 0) {
                printf("invalid audio embed dims (max_dim0=%d, dim1=%d)\n", max_dim0, embeds_dim1);
                return -1;
            }
            float16* chunk_embeds =
                (float16*)malloc((size_t)max_dim0 * (size_t)embeds_dim1 * sizeof(float16));
            if (!chunk_embeds) {
                printf("failed to alloc chunk_embeds buffer\n");
                return -1;
            }

            int total_valid = 0;
            printf("--> long audio: %d samples (%.3fs), splitting into %d chunks\n",
                   audio->num_frames, (float)audio->num_frames / 16000.0f, n_chunks);
            for (int c = 0; c < n_chunks; c++) {
                int offset = c * GEMMA4_MAX_CHUNK_SAMPLES;
                int len = audio->num_frames - offset;
                if (len > GEMMA4_MAX_CHUNK_SAMPLES) len = GEMMA4_MAX_CHUNK_SAMPLES;

                audio_buffer_t chunk;
                chunk.data         = audio->data + offset;
                chunk.num_frames   = len;
                chunk.num_channels = audio->num_channels;
                chunk.sample_rate  = audio->sample_rate;

                int chunk_valid = 0;
                printf("--> audio chunk %d/%d: samples [%d, %d) len=%d\n",
                       c + 1, n_chunks, offset, offset + len, len);
                ret = inference_gemma4_audio(&app_ctx->audio, &chunk, chunk_embeds, &chunk_valid);
                if (ret != 0) {
                    printf("inference_gemma4_audio failed on chunk %d! ret=%d\n", c, ret);
                    free(chunk_embeds);
                    return ret;
                }
                // Append the valid (front-loaded) rows of this chunk.
                memcpy(audio_embeds + (size_t)total_valid * (size_t)embeds_dim1,
                       chunk_embeds,
                       (size_t)chunk_valid * (size_t)embeds_dim1 * sizeof(float16));
                total_valid += chunk_valid;
            }
            free(chunk_embeds);
            n_valid_audio = total_valid;
            printf("--> long audio: concatenated %d valid audio tokens from %d chunks\n",
                   total_valid, n_chunks);
        }

        perf->audio_latency = getCurrentTimeUs() - start_us;
        tensor.audio.n_audio_tokens = n_valid_audio;
        printf("audio n_audio_tokens: %d\n", tensor.audio.n_audio_tokens);
    }

    if (app_ctx->enable_vision && tensor.image.n_image) {
        int64_t start_us = getCurrentTimeUs();
        if (image_embeds == NULL) {
            image_embeds = (float16*)tensor.image.image_embed;
        }
        if (image_embeds == NULL) {
            printf("image_embeds is NULL\n");
            return -1;
        }
        tensor.image.image_embed = image_embeds;
        ret = inference_gemma4_vision(&app_ctx->vision, image, image_embeds);
        perf->vision_latency = getCurrentTimeUs() - start_us;
        if (ret != 0) {
            printf("inference_gemma4_vision failed! ret=%d\n", ret);
            return ret;
        }
        printf("image n_image_tokens: %d\n", tensor.image.n_image_tokens);
    }

    printf("--> inference gemma4 llm model\n");
    ret = inference_gemma4_llm(&app_ctx->llm, tensor, max_new_tokens, perf);
    if (ret != 0) {
        printf("inference_gemma4_llm failed! ret=%d\n", ret);
        return ret;
    }

    return ret;
}
