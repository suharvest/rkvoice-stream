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

#ifndef _RKNN_DEMO_GEMMA4_LLM_H_
#define _RKNN_DEMO_GEMMA4_LLM_H_

#ifdef __cplusplus

#include <stdint.h>

#include "common.h"
#include "rknn3_api.h"

#define GEMMA4_DEFAULT_MAX_NEW_TOKENS 128

typedef struct {
  rknn3_context    rknn_ctx;
  rknn3_session*   rknn_sess;
  rknn3_tensor*    input_tensors;
  int              n_input_tensors;
  int*             llm_ext_input_indices;
  int              n_llm_ext_inputs;
  rknn3_llm_config llm_config;
} rknn_gemma4_llm_context;

int init_gemma4_llm(rknn_gemma4_llm_context* llm_ctx, const char* model_path, const char* weight_path,
                    rknn3_llm_param* params, int n_params, RKLLMCallback* callback, uint32_t core_mask);

int release_gemma4_llm(rknn_gemma4_llm_context* llm_ctx);

int inference_gemma4_llm(rknn_gemma4_llm_context* llm_ctx, rknn3_llm_multimodal_tensor tensor, int32_t max_new_tokens,
                         rknn_perf_metrics_t* perf);

#endif

#endif // _RKNN_DEMO_GEMMA4_LLM_H_
