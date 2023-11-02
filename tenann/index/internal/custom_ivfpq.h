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

struct CustomIvfPqSearchParameters : faiss::IVFPQSearchParameters {
  float error_scale;
  CustomIvfPqSearchParameters() : error_scale(0) {}
  ~CustomIvfPqSearchParameters() {}
};

struct CustomIvfPq : faiss::IndexIVFPQ {
  std::vector<std::vector<float>> reconstruction_errors;

  /// @brief Default search parameter used for range search.
  /// It can only be a float in [0, 1].
  /// The larger its value, the higher the recall, the lower the performance.
  /// When it is set to 1, the recall can reach 100%, but the number of wrong results
  /// will be greatly increased, and in the extreme case, all database vectors will be returned.
  float error_scale = 0;

  CustomIvfPq(faiss::Index* quantizer, size_t d, size_t nlist, size_t M, size_t nbits_per_idx);

  void add_core(idx_t n, const float* x, const idx_t* xids, const idx_t* precomputed_idx) override;

  /// same as add_core, also:
  /// - output 2nd level residuals if residuals_2 != NULL
  /// - accepts precomputed_idx = nullptr
  void custom_add_core_o(idx_t n, const float* x, const idx_t* xids, float* residuals_2,
                         const idx_t* precomputed_idx = nullptr);

  void range_search(idx_t n, const float* x, float radius, faiss::RangeSearchResult* result,
                    const faiss::SearchParameters* params = nullptr) const override;

  void custom_range_search_preassigned(idx_t nx, const float* x, float radius, const idx_t* keys,
                                       const float* coarse_dis, faiss::RangeSearchResult* result,
                                       bool store_pairs = false,
                                       const CustomIvfPqSearchParameters* params = nullptr,
                                       faiss::IndexIVFStats* stats = nullptr) const;

  faiss::InvertedListScanner* custom_get_InvertedListScanner(bool store_pairs,
                                                      const faiss::IDSelector* sel,
                                                      float error_scale) const;
};

}  // namespace tenann