/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#pragma once

#include "faiss/IndexIVFPQ.h"

namespace tenann {

struct CustomIvfPq : faiss::IndexIVFPQ {
  std::vector<std::vector<float>> reconstruction_errors;
  float error_scale = 0.1;  // 0~1，越大Recall越高，性能越低，为0的时候，代表忽略错误

  CustomIvfPq(faiss::Index* quantizer, size_t d, size_t nlist, size_t M, size_t nbits_per_idx);

  void add_core(idx_t n, const float* x, const idx_t* xids, const idx_t* precomputed_idx) override;

  /// same as add_core, also:
  /// - output 2nd level residuals if residuals_2 != NULL
  /// - accepts precomputed_idx = nullptr
  void custom_add_core_o(idx_t n, const float* x, const idx_t* xids, float* residuals_2,
                         const idx_t* precomputed_idx = nullptr);

  faiss::InvertedListScanner* get_InvertedListScanner(bool store_pairs,
                                                      const faiss::IDSelector* sel) const override;
};

}  // namespace tenann