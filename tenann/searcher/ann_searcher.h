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

#include <memory>
#include <vector>

#include "faiss/MetricType.h"
#include "tenann/common/seq_view.h"
#include "tenann/index/parameters.h"
#include "tenann/searcher/id_filter.h"
#include "tenann/searcher/searcher.h"

namespace tenann {

#define ANN_SEARCHER_QUERY_COUNT (1)

class AnnSearcher : public Searcher<AnnSearcher> {
 public:
  enum ResultOrder { kAscending = 0, kDescending = 1 };

  explicit AnnSearcher(const IndexMeta& meta);
  virtual ~AnnSearcher() override;

  T_FORBID_MOVE(AnnSearcher);
  T_FORBID_COPY_AND_ASSIGN(AnnSearcher);

  /**
   * @brief Approximate nearest neighbor search. Return both the qualified IDs and distances.
   *
   * @param query_vector     The query vector to search for.
   * @param k                The number of nearest neighbors to be returned.
   * @param result_id        A pointer to an array where the result ID will be stored. Should be of
   * @param result_distances A pointer to an array where the result distance will be stored.
   * size k.
   * @param id_filter        User-defined rowid filter.
   */
  virtual void AnnSearch(PrimitiveSeqView query_vector, int64_t k, int64_t* result_id,
                         uint8_t* result_distances, const IdFilter* id_filter = nullptr) = 0;

  /// @brief Approximate nearest neighbor search. Return only the IDs without the distances.
  virtual void AnnSearch(PrimitiveSeqView query_vector, int64_t k, int64_t* result_id,
                         const IdFilter* id_filter = nullptr) = 0;

  /**
   * @brief Range search. Return both the qualified IDs and distances.
   *
   * @param query_vector     The query vector to search for.
   * @param range            Range threshold.
   * @param limit            The maximum number of results to return. Set -1 to disable limit.
   * @param result_order     The result_order of results: asending, desending, or unordered.
   * @param result_ids       A pointer to an vector where the result IDs will be stored. The given
   * vector would be resized by the function to accommodate the results.
   * @param result_distances A pointer to an vector where the result distances will be stored. The
   * given vector would be resized by the function to accommodate the results.
   * @param id_filter        User-defined rowid filter.
   */
  virtual void RangeSearch(PrimitiveSeqView query_vector, float range, int64_t limit,
                           ResultOrder result_order, std::vector<int64_t>* result_ids,
                           std::vector<float>* result_distances,
                           const IdFilter* id_filter = nullptr);

  /// @brief Range search. Return only the IDs without the distances.
  virtual void RangeSearch(PrimitiveSeqView query_vector, float range, int64_t limit,
                           ResultOrder result_order, std::vector<int64_t>* result_ids,
                           const IdFilter* id_filter = nullptr);

 protected:
  /**
   * @brief Returns the query to search with, normalized into `scratch` when the index needs that
   * and nothing else will do it.
   *
   * Cosine is defined on directions, so a query that is not unit length would otherwise scale every
   * score. An index built with is_vector_normed=false carries a normalizing pre-transform that
   * covers the query too; one built with is_vector_normed=true does not -- there the caller
   * promised unit-norm input and only the write path verifies that promise.
   *
   * A zero vector is left alone rather than turned into NaNs, which keeps its inner product at 0.
   */
  const float* PrepareCosineQuery(const float* query, size_t dim,
                                  std::vector<float>* scratch) const;

  /**
   * @brief Validate a logical cosine threshold and convert it for the physical metric.
   */
  float PrepareCosineRange(float range, faiss::MetricType physical_metric) const;

  /**
   * @brief Convert physical cosine scores to the logical representation and clamp valid results.
   *
   * Faiss pads an under-filled top-k result with id -1 and metric-specific distance sentinels.
   * Those padding scores are intentionally left untouched; callers must use the ID to determine
   * whether a result slot is valid.
   */
  void FinalizeScores(const int64_t* ids, float* scores, size_t n,
                      faiss::MetricType physical_metric) const;

  VectorIndexCommonParams common_params_;
};

using AnnSearcherRef = std::shared_ptr<AnnSearcher>;

}  // namespace tenann
