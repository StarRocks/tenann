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

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include <memory>
#include <queue>

#include "tenann/common/logging.h"
#include "tenann/common/typed_seq_view.h"
#include "tenann/searcher/ann_searcher.h"
#include "tenann/store/index_type.h"
#include "tenann/util/distance_util.h"

namespace tenann {

namespace util {

using dist_t = float;
using idx_t = int64_t;

struct DistanceComputer {
  virtual ~DistanceComputer() = default;

  dist_t operator()(const dist_t* v1, const dist_t* v2, size_t dim) const {
    return Apply(v1, v2, dim);
  }

  virtual dist_t Apply(const dist_t* v1, const dist_t* v2, size_t dim) const = 0;
};

struct EuclideanDistance final : DistanceComputer {
  ~EuclideanDistance() override = default;

  dist_t Apply(const dist_t* v1, const dist_t* v2, size_t dim) const override {
    dist_t sum = 0.0;
    for (size_t i = 0; i < dim; i++) {
      dist_t diff = *(v2 + i) - *(v1 + i);
      sum += diff * diff;
    }

    // return std::sqrt(sum); faiss not sqrt
    return sum;
  }
};

struct CosineSimilarity final : DistanceComputer {
  ~CosineSimilarity() override = default;

  dist_t Apply(const dist_t* v1, const dist_t* v2, size_t dim) const override {
    dist_t dot_product = 0.0f;
    dist_t norm_v1 = 0.0f;
    dist_t norm_v2 = 0.0f;

    for (size_t i = 0; i < dim; ++i) {
      dot_product += v1[i] * v2[i];
      norm_v1 += v1[i] * v1[i];
      norm_v2 += v2[i] * v2[i];
    }

    if (norm_v1 == 0.0f || norm_v2 == 0.0f) {
      return 0.0f;
    }

    return dot_product / (sqrtf(norm_v1) * sqrtf(norm_v2));
  }
};

struct MaxFirst {
  constexpr bool operator()(std::pair<dist_t, idx_t> const& a,
                            std::pair<dist_t, idx_t> const& b) const noexcept {
    return a.first > b.first;
  }
};

struct MinFirst {
  constexpr bool operator()(std::pair<dist_t, idx_t> const& a,
                            std::pair<dist_t, idx_t> const& b) const noexcept {
    return a.first < b.first;
  }
};

struct RangeFilter {
  dist_t threshold;
  bool asending;

  bool IsQualified(dist_t d) {
    if (asending) {
      return d <= threshold;
    } else {
      return d >= threshold;
    }
  }
};

namespace detail {
template <typename Comparator>
struct BruteForceAnnSearcher {
  void Search(size_t dim, const SeqView& base_col, const uint8_t* null_flags, const int64_t* rowids,
              const SeqView& query_col, MetricType metric_type, int64_t k, int64_t* result_ids,
              dist_t* result_distances) {
    T_CHECK(!(null_flags != nullptr && rowids == nullptr));
    T_CHECK(metric_type == MetricType::kL2Distance || metric_type == MetricType::kCosineSimilarity);

    T_CHECK(base_col.seq_view_type == SeqViewType::kArraySeqView ||
            base_col.seq_view_type == SeqViewType::kVlArraySeqView);
    T_CHECK(!(base_col.seq_view_type == SeqViewType::kArraySeqView &&
              base_col.seq_view.array_seq_view.elem_type != PrimitiveType::kFloatType));
    T_CHECK(!(base_col.seq_view_type == SeqViewType::kVlArraySeqView &&
              base_col.seq_view.vl_array_seq_view.elem_type != PrimitiveType::kFloatType));

    T_CHECK(query_col.seq_view_type == SeqViewType::kArraySeqView ||
            query_col.seq_view_type == SeqViewType::kVlArraySeqView);
    T_CHECK(!(query_col.seq_view_type == SeqViewType::kArraySeqView &&
              query_col.seq_view.array_seq_view.elem_type != PrimitiveType::kFloatType));
    T_CHECK(!(query_col.seq_view_type == SeqViewType::kVlArraySeqView &&
              query_col.seq_view.vl_array_seq_view.elem_type != PrimitiveType::kFloatType));

    std::unique_ptr<DistanceComputer> distance_computer = std::make_unique<EuclideanDistance>();
    if (metric_type == MetricType::kCosineSimilarity) {
      distance_computer = std::make_unique<util::CosineSimilarity>();
    }

    // compute distances
    auto base_iter = base_col.seq_view_type == SeqViewType::kArraySeqView
                         ? TypedSliceIterator<dist_t>(base_col.seq_view.array_seq_view)
                         : TypedSliceIterator<dist_t>(base_col.seq_view.vl_array_seq_view);

    auto query_iter = query_col.seq_view_type == SeqViewType::kArraySeqView
                          ? TypedSliceIterator<dist_t>(query_col.seq_view.array_seq_view)
                          : TypedSliceIterator<dist_t>(query_col.seq_view.vl_array_seq_view);

    // sort
    query_iter.ForEach([&](idx_t query_idx, const dist_t* query_data, idx_t query_length) {
      std::priority_queue<std::pair<dist_t, idx_t>, std::vector<std::pair<dist_t, idx_t>>,
                          Comparator>
          results;
      base_iter.ForEach([&](idx_t base_idx, const dist_t* base_data, idx_t base_length) {
        T_CHECK_EQ(base_length, dim);
        T_CHECK_EQ(query_length, dim);

        if (null_flags == nullptr || (null_flags != nullptr && null_flags[base_idx] == 0)) {
          auto distance = distance_computer->Apply(base_data, query_data, dim);
          if (rowids != nullptr) {
            results.emplace(distance, rowids[base_idx]);
          } else {
            results.emplace(distance, base_idx);
          }
        }

        while (results.size() > k) {
          results.pop();
        }
      });

      T_DCHECK_GE(results.size(), k);
      if (results.size() < k) {
        for (int j = results.size(); j < k; j++) {
          result_distances[query_idx * k + j] = 0;
          result_ids[query_idx * k + j] = -1;
        }
      }

      for (int j = results.size() - 1; j >= 0; j--) {
        if (!results.empty()) {
          auto [distance, id] = results.top();
          result_distances[query_idx * k + j] = distance;
          result_ids[query_idx * k + j] = id;
          results.pop();
        }
      }
    });
  }
};

struct BruteForceRangeSearcher {
  void Search(MetricType metric_type, size_t dim, const SeqView& base_col,
              const uint8_t* null_flags, const int64_t* rowids, PrimitiveSeqView query_vector,
              float range, int64_t limit, AnnSearcher::ResultOrder result_order,
              std::vector<int64_t>* result_ids, std::vector<float>* result_distances,
              const IdFilter* id_filter = nullptr) {
    T_CHECK(!(null_flags != nullptr && rowids == nullptr));
    T_CHECK(metric_type == MetricType::kL2Distance || metric_type == MetricType::kCosineSimilarity);
    T_CHECK(!(metric_type == MetricType::kL2Distance &&
              result_order != AnnSearcher::ResultOrder::kAscending));
    T_CHECK(!(metric_type == MetricType::kCosineSimilarity &&
              result_order != AnnSearcher::ResultOrder::kDescending));

    T_CHECK(base_col.seq_view_type == SeqViewType::kArraySeqView ||
            base_col.seq_view_type == SeqViewType::kVlArraySeqView);
    T_CHECK(!(base_col.seq_view_type == SeqViewType::kArraySeqView &&
              base_col.seq_view.array_seq_view.elem_type != PrimitiveType::kFloatType));
    T_CHECK(!(base_col.seq_view_type == SeqViewType::kVlArraySeqView &&
              base_col.seq_view.vl_array_seq_view.elem_type != PrimitiveType::kFloatType));

    T_CHECK(query_vector.elem_type == PrimitiveType::kFloatType);
    T_CHECK(query_vector.size == dim);

    bool ascending = result_order == AnnSearcher::ResultOrder::kAscending;
    std::unique_ptr<DistanceComputer> distance_computer = std::make_unique<EuclideanDistance>();
    if (metric_type == MetricType::kCosineSimilarity) {
      distance_computer = std::make_unique<util::CosineSimilarity>();
    }

    RangeFilter filter{.threshold = range, .asending = ascending};
    auto base_iter = base_col.seq_view_type == SeqViewType::kArraySeqView
                         ? TypedSliceIterator<dist_t>(base_col.seq_view.array_seq_view)
                         : TypedSliceIterator<dist_t>(base_col.seq_view.vl_array_seq_view);

    auto* query_data = (const dist_t*)query_vector.data;

    base_iter.ForEach([&](idx_t base_idx, const dist_t* base_data, idx_t base_length) {
      T_CHECK_EQ(base_length, dim);

      if (null_flags == nullptr || (null_flags != nullptr && null_flags[base_idx] == 0)) {
        auto distance = distance_computer->Apply(base_data, query_data, dim);

        if (filter.IsQualified(distance)) {
          result_distances->push_back(distance);
          if (rowids != nullptr) {
            result_ids->push_back(rowids[base_idx]);
          } else {
            result_ids->push_back(base_idx);
          }
        }
      }
    });

    // sort results
    if (limit > 0) {
      ReserveTopK(result_ids, result_distances, limit, ascending);
    } else {
      ReserveTopK(result_ids, result_distances, result_ids->size(), ascending);
    }
  }
};

}  // namespace detail

inline void BruteForceTopKSearch(size_t dim, const SeqView& base_col, const uint8_t* null_flags,
                                 const int64_t* rowids, const SeqView& query_col,
                                 MetricType metric_type, int64_t k, int64_t* result_ids,
                                 dist_t* result_distances) {
  if (metric_type == MetricType::kL2Distance) {
    detail::BruteForceAnnSearcher<MinFirst> searcher;
    searcher.Search(dim, base_col, null_flags, rowids, query_col, metric_type, k, result_ids,
                    result_distances);
  } else if (metric_type == MetricType::kCosineSimilarity) {
    detail::BruteForceAnnSearcher<MaxFirst> searcher;
    searcher.Search(dim, base_col, null_flags, rowids, query_col, metric_type, k, result_ids,
                    result_distances);
  } else {
    T_LOG(ERROR) << "unsupported metric type";
  }
}

inline void BruteForceRangeSearch(MetricType metric_type, size_t dim, const SeqView& base_col,
                                  const uint8_t* null_flags, const int64_t* rowids,
                                  PrimitiveSeqView query_vector, float range, int64_t limit,
                                  AnnSearcher::ResultOrder result_order,
                                  std::vector<int64_t>* result_ids,
                                  std::vector<float>* result_distances,
                                  const IdFilter* id_filter = nullptr) {
  detail::BruteForceRangeSearcher searcher;
  searcher.Search(metric_type, dim, base_col, null_flags, rowids, query_vector, range, limit,
                  result_order, result_ids, result_distances, id_filter);
};

}  // namespace util
}  // namespace tenann