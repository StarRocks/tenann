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

#include "tenann/searcher/faiss_hnsw_ann_searcher.h"

#include <algorithm>

#include "faiss/IndexHNSW.h"
#include "faiss/IndexIDMap.h"
#include "faiss/impl/FaissException.h"
#include "faiss/impl/HNSW.h"
#include "faiss_hnsw_ann_searcher.h"
#include "tenann/common/logging.h"
#include "tenann/index/internal/faiss_index_util.h"
#include "tenann/index/parameter_serde.h"
#include "tenann/searcher/internal/id_filter_adapter.h"
#include "tenann/store/index_meta.h"
#include "tenann/util/distance_util.h"

namespace tenann {

namespace detail {
using namespace faiss;
using MinimaxHeap = HNSW::MinimaxHeap;
using storage_idx_t = HNSW::storage_idx_t;

/** Copied from faiss/IndexHNSW.cpp */
DistanceComputer* storage_distance_computer(const faiss::Index* storage) {
  if (storage->metric_type == METRIC_INNER_PRODUCT) {
    T_LOG(ERROR) << "inner product is not supported now";
    return nullptr;
  } else {
    return storage->get_distance_computer();
  }
}

/** Copied from faiss/impl/HNSW.cpp */
/// greedily update a nearest vector at a given level
void greedy_update_nearest(const HNSW& hnsw, DistanceComputer& qdis, int level,
                           storage_idx_t& nearest, float& d_nearest) {
  for (;;) {
    storage_idx_t prev_nearest = nearest;

    size_t begin, end;
    hnsw.neighbor_range(nearest, level, &begin, &end);
    for (size_t i = begin; i < end; i++) {
      storage_idx_t v = hnsw.neighbors[i];
      if (v < 0) break;
      float dis = qdis(v);
      if (dis < d_nearest) {
        nearest = v;
        d_nearest = dis;
      }
    }
    if (nearest == prev_nearest) {
      return;
    }
  }
}

/** Ported from faiss/impl/HNSW.cpp */
/**
 * @brief Range search based on HNSW without result number limit.
 *
 * Assuming the number of results for a range query is k, this
 * algorithm is equivalent to an efficient Top-$k$ search. Therefore, this algorithm is suitable for
 * cases where the number of results is relatively small (i.e., radius is small). When the number of
 * results is very large, the efficiency of this algorithm may deteriorate rapidly, even worse than
 * brute-force search. Therefore, you should carefully use this algorithm based on your specific
 * application scenario.
 *
 * @param hnsw Faiss HNSW structure
 * @param qdis The distance computer
 * @param radius Distance limit
 * @param k Limit of the results, if [[k]] is given as a positive number, only top-k results
 * will be preserved. If [[k]] is given as zero or a negative number,
 * @param I Result indices
 * @param D Result distances
 * @param candidates Candidates that the search starts with
 * @param vt Visit table
 * @param level Graph level to search
 * @param result Range search results
 * @param params Search params
 * @return int Number of results
 */
void HnswRangeSearchFromCandidates(const HNSW& hnsw, DistanceComputer& qdis, float radius,
                                   std::priority_queue<HNSW::Node>* results,
                                   MinimaxHeap& candidates, VisitedTable& vt, int level,
                                   const SearchParametersHNSW* params = nullptr) {
  int ndis = 0;

  // can be overridden by search params
  int efSearch = params ? params->efSearch : hnsw.efSearch;
  const IDSelector* sel = params ? params->sel : nullptr;

  for (int i = 0; i < candidates.size(); i++) {
    storage_idx_t v1 = candidates.ids[i];
    float d = candidates.dis[i];
    FAISS_ASSERT(v1 >= 0);
    if (!sel || sel->is_member(v1)) {
      if (d <= radius) {
        results->emplace(d, v1);
      }
    }
    vt.set(v1);
  }

  int nstep = 0;

  while (candidates.size() > 0) {
    float d0 = 0;
    int v0 = candidates.pop_min(&d0);

    size_t begin, end;
    hnsw.neighbor_range(v0, level, &begin, &end);

    for (size_t j = begin; j < end; j++) {
      int v1 = hnsw.neighbors[j];
      if (v1 < 0) break;
      if (vt.get(v1)) {
        continue;
      }
      vt.set(v1);
      ndis++;
      float d = qdis(v1);
      if (!sel || sel->is_member(v1)) {
        if (d <= radius) {
          results->emplace(d, v1);
        }
      }
      candidates.push(v1, d);
    }

    nstep++;
    if (nstep > efSearch) {
      break;
    }
  }
}

void IndexHnswRangeSearch(const IndexHNSW& index, idx_t n, const float* x, float radius,
                          int64_t limit, std::vector<idx_t>* result_ids,
                          std::vector<float>* result_distances,
                          const SearchParametersHNSW* params = nullptr) {
  T_CHECK_EQ(n, 1) << "batch search is not supported now, only 1 query vector is allowed";
  int64_t efSearch = params ? params->efSearch : index.hnsw.efSearch;
  int64_t ef = std::max(efSearch, limit);

  if (index.hnsw.entry_point == -1) {
    return;
  }

  VisitedTable vt(index.ntotal);

  if (limit > 0) {  // search top-ef nearest neighbors first, then perform post filtering based on
                    // the returned distances
    result_ids->resize(ef);
    result_distances->resize(ef);
    index.search(n, x, ef, result_distances->data(), result_ids->data(), params);

    idx_t n = 0;
    for (idx_t i = 0; i < ef; i++) {
      if ((*result_distances)[i] <= radius) {
        n += 1;
      } else {
        break;
      }
    }
    auto resize = std::min(n, limit);
    result_ids->resize(resize);
    result_distances->resize(resize);
  } else {  // range search without result limit
    if (index.hnsw.upper_beam != 1) {
      T_LOG(WARNING)
          << "upper_beam is set, but it takes no effects on hnsw range search without limit";
    }

    if (!index.hnsw.search_bounded_queue) {
      T_LOG(WARNING) << "search_bounded_queue is set to false, but it takes no effects on hnsw "
                        "range search without limit";
    }

    DistanceComputer* p_dis = storage_distance_computer(index.storage);
    ScopeDeleter1<DistanceComputer> del(p_dis);
    auto& dis = *p_dis;
    dis.set_query(x);

    // greedy search on upper levels
    storage_idx_t nearest = index.hnsw.entry_point;
    float d_nearest = dis(nearest);
    for (int level = index.hnsw.max_level; level >= 1; level--) {
      greedy_update_nearest(index.hnsw, dis, level, nearest, d_nearest);
    }

    MinimaxHeap candidates(ef);
    candidates.push(nearest, d_nearest);

    std::priority_queue<HNSW::Node> result_queue;
    HnswRangeSearchFromCandidates(index.hnsw, dis, radius, &result_queue, candidates, vt, 0,
                                  params);
    result_ids->resize(result_queue.size());
    result_distances->resize(result_queue.size());

    int i = result_queue.size() - 1;
    while (!result_queue.empty()) {
      auto [d, id] = result_queue.top();
      result_queue.pop();
      (*result_distances)[i] = d;
      (*result_ids)[i] = id;
      i -= 1;
    }

    vt.advance();
  }
  return;
}

}  // namespace detail

FaissHnswAnnSearcher::FaissHnswAnnSearcher(const IndexMeta& meta) : AnnSearcher(meta) {
  FetchParameters(meta, &search_params_);
}

FaissHnswAnnSearcher::~FaissHnswAnnSearcher() = default;

void FaissHnswAnnSearcher::AnnSearch(PrimitiveSeqView query_vector, int k, int64_t* result_id,
                                     const IdFilter* id_filter) {
  std::vector<float> distances(k);
  AnnSearch(query_vector, k, result_id, reinterpret_cast<uint8_t*>(distances.data()), id_filter);
}

void FaissHnswAnnSearcher::AnnSearch(PrimitiveSeqView query_vector, int k, int64_t* result_ids,
                                     uint8_t* result_distances, const IdFilter* id_filter) {
  T_CHECK_NOTNULL(index_ref_);

  T_CHECK_EQ(index_ref_->index_type(), IndexType::kFaissHnsw);
  T_CHECK_EQ(query_vector.elem_type, PrimitiveType::kFloatType);

  faiss::SearchParametersHNSW faiss_search_parameters;
  faiss_search_parameters.efSearch = search_params_.efSearch;
  faiss_search_parameters.check_relative_distance = search_params_.check_relative_distance;
  std::shared_ptr<IdFilterAdapter> id_filter_adapter;
  if (id_filter) {
    if (faiss_id_map_ != nullptr) {
      id_filter_adapter = IdFilterAdapterFactory::CreateIdFilterAdapter(
          id_filter, &reinterpret_cast<const faiss::IndexIDMap*>(faiss_id_map_)->id_map);
    } else {
      id_filter_adapter = IdFilterAdapterFactory::CreateIdFilterAdapter(id_filter);
    }
    faiss_search_parameters.sel = id_filter_adapter.get();
  }

  VLOG(VERBOSE_DEBUG) << "efSearch: " << faiss_search_parameters.efSearch
                      << ", check_relative_distance: "
                      << faiss_search_parameters.check_relative_distance;

  // transform the query vector first if a pre-transform is set
  const float* x = reinterpret_cast<const float*>(query_vector.data);
  if (faiss_transform_ != nullptr) {
    const float* xt = reinterpret_cast<const faiss::IndexPreTransform*>(faiss_transform_)
                          ->apply_chain(ANN_SEARCHER_QUERY_COUNT, x);
    faiss::ScopeDeleter<float> del(xt == x ? nullptr : xt);
    // search through the transformed vector
    reinterpret_cast<const faiss::IndexHNSW*>(faiss_hnsw_)
        ->search(ANN_SEARCHER_QUERY_COUNT, xt, k, reinterpret_cast<float*>(result_distances),
                 result_ids, &faiss_search_parameters);
  } else {
    reinterpret_cast<const faiss::IndexHNSW*>(faiss_hnsw_)
        ->search(ANN_SEARCHER_QUERY_COUNT, x, k, reinterpret_cast<float*>(result_distances),
                 result_ids, &faiss_search_parameters);
  }

  if (faiss_id_map_ != nullptr) {
    int64_t* li = result_ids;
    for (int64_t i = 0; i < ANN_SEARCHER_QUERY_COUNT * k; i++) {
      li[i] = li[i] < 0 ? li[i]
                        : reinterpret_cast<const faiss::IndexIDMap*>(faiss_id_map_)->id_map[li[i]];
    }
  }

  if (common_params_.metric_type == MetricType::kCosineSimilarity) {
    auto distances = reinterpret_cast<float*>(result_distances);
    L2DistanceToCosineSimilarity(distances, distances, k);
  }
}

void FaissHnswAnnSearcher::RangeSearch(PrimitiveSeqView query_vector, float range, int64_t limit,
                                       ResultOrder result_order, std::vector<int64_t>* result_ids,
                                       std::vector<float>* result_distances,
                                       const IdFilter* id_filter) {
  T_CHECK_NOTNULL(index_ref_);

  T_CHECK_EQ(index_ref_->index_type(), IndexType::kFaissHnsw);
  T_CHECK_EQ(query_vector.elem_type, PrimitiveType::kFloatType);
  T_CHECK_NE(common_params_.metric_type, MetricType::kInnerProduct)
      << "Range search is currently not supported for inner product metric.";

  float radius = range;
  if (common_params_.metric_type == MetricType::kCosineSimilarity) {
    radius = CosineSimilarityThresholdToL2Distance(range);
    T_CHECK(result_order == ResultOrder::kDescending)
        << "only descending order is allowed for range search results based on cosine similarity";
  } else if (common_params_.metric_type == MetricType::kL2Distance) {
    T_CHECK(result_order == ResultOrder::kAscending)
        << "only ascending order is allowed for range search with l2 distance";
  } else {
    T_LOG(ERROR) << "using unsupported distance metric, hnsw range search only supports l2 "
                    "distance and cosine similarity";
  }

  faiss::SearchParametersHNSW faiss_search_parameters;
  faiss_search_parameters.efSearch = search_params_.efSearch;
  faiss_search_parameters.check_relative_distance = search_params_.check_relative_distance;
  std::shared_ptr<IdFilterAdapter> id_filter_adapter;

  VLOG(VERBOSE_DEBUG) << "efSearch: " << faiss_search_parameters.efSearch
                      << ", check_relative_distance: "
                      << faiss_search_parameters.check_relative_distance << ", range: " << range
                      << ", radius: " << radius << ", limit: " << limit
                      << ", result_order: " << result_order;

  if (id_filter) {
    if (faiss_id_map_ != nullptr) {
      id_filter_adapter = IdFilterAdapterFactory::CreateIdFilterAdapter(
          id_filter, &reinterpret_cast<const faiss::IndexIDMap*>(faiss_id_map_)->id_map);
    } else {
      id_filter_adapter = IdFilterAdapterFactory::CreateIdFilterAdapter(id_filter);
    }
    faiss_search_parameters.sel = id_filter_adapter.get();
  }

  // Transform the query vector first if a pre-transform is set
  const float* x = reinterpret_cast<const float*>(query_vector.data);
  if (faiss_transform_ != nullptr) {
    const float* xt = reinterpret_cast<const faiss::IndexPreTransform*>(faiss_transform_)
                          ->apply_chain(ANN_SEARCHER_QUERY_COUNT, x);
    faiss::ScopeDeleter<float> del(xt == x ? nullptr : xt);
    // Search with the transformed vector
    detail::IndexHnswRangeSearch(*reinterpret_cast<const faiss::IndexHNSW*>(faiss_hnsw_),
                                 ANN_SEARCHER_QUERY_COUNT, xt, radius, limit, result_ids,
                                 result_distances, &faiss_search_parameters);
  } else {
    // Search with the raw vector
    detail::IndexHnswRangeSearch(*reinterpret_cast<const faiss::IndexHNSW*>(faiss_hnsw_),
                                 ANN_SEARCHER_QUERY_COUNT, x, radius, limit, result_ids,
                                 result_distances, &faiss_search_parameters);
  }

  if (faiss_id_map_ != nullptr) {
    int64_t* li = result_ids->data();
    for (int64_t i = 0; i < ANN_SEARCHER_QUERY_COUNT * result_ids->size(); i++) {
      li[i] = li[i] < 0 ? li[i]
                        : reinterpret_cast<const faiss::IndexIDMap*>(faiss_id_map_)->id_map[li[i]];
    }
  }

  if (common_params_.metric_type == MetricType::kCosineSimilarity) {
    auto distances = reinterpret_cast<float*>(result_distances->data());
    L2DistanceToCosineSimilarity(distances, distances, result_distances->size());
  }
}

void FaissHnswAnnSearcher::OnSearchParamItemChange(const std::string& key, const json& value) {
  if (key == FaissHnswSearchParams::efSearch_key) {
    value.is_number_integer() && (search_params_.efSearch = value.get<int>());
    // T_LOG(INFO) << "here:: " << key;
  } else if (key == FaissHnswSearchParams::check_relative_distance_key) {
    value.is_boolean() && (search_params_.check_relative_distance = value.get<bool>());
    // T_LOG(INFO) << "here:: " << key;
  } else {
    T_LOG(ERROR) << "Unsupport search parameter: " << key;
  }
};

void FaissHnswAnnSearcher::OnSearchParamsChange(const json& value) {
  for (auto it = value.begin(); it != value.end(); ++it) {
    OnSearchParamItemChange(it.key(), it.value());
  }
}

void FaissHnswAnnSearcher::OnIndexLoaded() {
  // fetch and check faiss index here
  auto faiss_index = static_cast<faiss::Index*>(index_ref_->index_raw());
  auto [id_map, transform, hnsw] = faiss_util::UnpackHnsw(faiss_index, common_params_);
  faiss_id_map_ = id_map;
  faiss_transform_ = transform;
  faiss_hnsw_ = hnsw;
}

}  // namespace tenann
