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
#include "faiss/IndexPreTransform.h"

namespace tenann {

using faiss::idx_t;

/// Distance kernels for 8-bit PQ codes. `sim_table` is a row-major
/// (pq_m x pq_ksub) table of precomputed sub-distances, so code m indexes row m.
/// The kernels are declared here so that a test can compare them against each other;
/// production code should call Pq8DistanceSingleCode.
namespace ivfpq_simd {

float Pq8DistanceSingleCodeGeneric(const uint8_t* code, const float* sim_table, size_t pq_m,
                                   size_t pq_ksub);

#if defined(__x86_64__)
/// True when the running CPU supports AVX2. Detected once.
bool Avx2Supported();

float Pq8DistanceSingleCodeAvx2(const uint8_t* code, const float* sim_table, size_t pq_m,
                                size_t pq_ksub);
#endif  // __x86_64__

/// Picks the widest kernel the running CPU supports. Inline so the branch folds into
/// the scan loop rather than adding a second call per code.
inline float Pq8DistanceSingleCode(const uint8_t* code, const float* sim_table, size_t pq_m,
                                   size_t pq_ksub) {
#if defined(__x86_64__)
  if (Avx2Supported()) {
    return Pq8DistanceSingleCodeAvx2(code, sim_table, pq_m, pq_ksub);
  }
#endif
  return Pq8DistanceSingleCodeGeneric(code, sim_table, pq_m, pq_ksub);
}

}  // namespace ivfpq_simd

namespace detail {

template <class C>
inline bool IsWithinRangeInclusive(float distance, float radius) {
  // Keep the boundary inclusive without treating unordered NaN values as matches.
  return C::cmp(radius, distance) || distance == radius;
}

}  // namespace detail

struct IndexIvfPqSearchParameters : faiss::IVFPQSearchParameters {
  float range_search_confidence;
  IndexIvfPqSearchParameters() : range_search_confidence(0) {}
  ~IndexIvfPqSearchParameters() {}
};

struct IndexIvfPq : faiss::IndexIVFPQ {
  IndexIvfPq();
  ~IndexIvfPq() override;

  std::vector<std::vector<float>> reconstruction_errors;

  /// @brief Default search parameter used for range search.
  /// It can only be a float in [0, 1].
  /// The larger its value, the higher the recall, the lower the performance.
  /// When it is set to 1, the recall can reach 100%, but the number of wrong results
  /// will be greatly increased, and in the extreme case, all database vectors will be returned.
  float range_search_confidence = 0;

  IndexIvfPq(faiss::Index* quantizer, size_t d, size_t nlist, size_t M, size_t nbits_per_idx,
             faiss::MetricType metric = faiss::METRIC_L2);

  void add_core(idx_t n, const float* x, const idx_t* xids, const idx_t* precomputed_idx,
                void* inverted_list_context = nullptr) override;

  /// same as add_core, also:
  /// - output 2nd level residuals if residuals_2 != NULL
  /// - accepts precomputed_idx = nullptr
  void custom_add_core_o(idx_t n, const float* x, const idx_t* xids, float* residuals_2,
                         const idx_t* precomputed_idx = nullptr,
                         void* inverted_list_context = nullptr);

  void range_search(idx_t n, const float* x, float radius, faiss::RangeSearchResult* result,
                    const faiss::SearchParameters* params = nullptr) const override;

  void custom_range_search_preassigned(idx_t nx, const float* x, float radius, const idx_t* keys,
                                       const float* coarse_dis, faiss::RangeSearchResult* result,
                                       bool store_pairs = false,
                                       const IndexIvfPqSearchParameters* params = nullptr,
                                       faiss::IndexIVFStats* stats = nullptr) const;

  faiss::InvertedListScanner* custom_get_InvertedListScanner(bool store_pairs,
                                                             const faiss::IDSelector* sel,
                                                             float range_search_confidence) const;
};

}  // namespace tenann
