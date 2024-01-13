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

#include "tenann/searcher/faiss_ivf_pq_ann_searcher.h"

#include <algorithm>
#include <queue>

#include "faiss/IndexIVFPQ.h"
#include "faiss/impl/AuxIndexStructures.h"
#include "faiss_ivf_pq_ann_searcher.h"
#include "tenann/common/logging.h"
#include "tenann/index/internal/faiss_index_util.h"
#include "tenann/index/internal/index_ivfpq.h"
#include "tenann/index/parameter_serde.h"
#include "tenann/index/parameters.h"
#include "tenann/searcher/internal/id_filter_adapter.h"
#include "tenann/util/distance_util.h"
namespace tenann {

FaissIvfPqAnnSearcher::FaissIvfPqAnnSearcher(const IndexMeta& meta) : AnnSearcher(meta) {
  FetchParameters(meta, &search_params_);
}

FaissIvfPqAnnSearcher::~FaissIvfPqAnnSearcher() = default;

void FaissIvfPqAnnSearcher::AnnSearch(PrimitiveSeqView query_vector, int64_t k, int64_t* result_id,
                                      const IdFilter* id_filter) {
  std::vector<float> distances(k);
  AnnSearch(query_vector, k, result_id, reinterpret_cast<uint8_t*>(distances.data()), id_filter);
}

void FaissIvfPqAnnSearcher::AnnSearch(PrimitiveSeqView query_vector, int64_t k, int64_t* result_ids,
                                      uint8_t* result_distances, const IdFilter* id_filter) {
  try {
    T_CHECK_NOTNULL(index_ref_);

    T_CHECK_EQ(index_ref_->index_type(), IndexType::kFaissIvfPq);
    T_CHECK_EQ(query_vector.elem_type, PrimitiveType::kFloatType);

    auto faiss_index = static_cast<faiss::Index*>(index_ref_->index_raw());

    faiss::IVFPQSearchParameters faiss_search_parameters;
    faiss_search_parameters.nprobe = search_params_.nprobe;
    faiss_search_parameters.max_codes = search_params_.max_codes;
    faiss_search_parameters.polysemous_ht = search_params_.polysemous_ht;
    faiss_search_parameters.scan_table_threshold = search_params_.scan_table_threshold;
    std::shared_ptr<IdFilterAdapter> id_filter_adapter;
    if (id_filter) {
      id_filter_adapter = IdFilterAdapterFactory::CreateIdFilterAdapter(id_filter);
      faiss_search_parameters.sel = id_filter_adapter.get();
    }

    VLOG(VERBOSE_DEBUG) << "nprobe: " << faiss_search_parameters.nprobe;

    faiss_index->search(ANN_SEARCHER_QUERY_COUNT, reinterpret_cast<const float*>(query_vector.data),
                        k, reinterpret_cast<float*>(result_distances), result_ids,
                        &faiss_search_parameters);

    if (common_params_.metric_type == MetricType::kCosineSimilarity) {
      auto distances = reinterpret_cast<float*>(result_distances);
      L2DistanceToCosineSimilarity(distances, distances, k);
    }
  }
  CATCH_FAISS_ERROR
}

void FaissIvfPqAnnSearcher::RangeSearch(PrimitiveSeqView query_vector, float range, int64_t limit,
                                        ResultOrder result_order, std::vector<int64_t>* result_ids,
                                        std::vector<float>* result_distances,
                                        const IdFilter* id_filter) {
  try {
    // TODO: add desending order support
    // T_CHECK(result_order != ResultOrder::kDescending) << "descending order not implemented";
    T_CHECK_NOTNULL(index_ref_);

    T_CHECK_EQ(index_ref_->index_type(), IndexType::kFaissIvfPq);
    T_CHECK_EQ(query_vector.elem_type, PrimitiveType::kFloatType);
    T_CHECK_NE(common_params_.metric_type, MetricType::kInnerProduct)
        << "Range search is currently not supported for inner product metric.";

    auto faiss_index = static_cast<const faiss::Index*>(index_ref_->index_raw());

    IndexIvfPqSearchParameters dynamic_search_parameters;
    dynamic_search_parameters.nprobe = search_params_.nprobe;
    dynamic_search_parameters.max_codes = search_params_.max_codes;
    dynamic_search_parameters.polysemous_ht = search_params_.polysemous_ht;
    dynamic_search_parameters.scan_table_threshold = search_params_.scan_table_threshold;
    dynamic_search_parameters.range_search_confidence = search_params_.range_search_confidence;
    std::shared_ptr<IdFilterAdapter> id_filter_adapter;
    if (id_filter) {
      id_filter_adapter = IdFilterAdapterFactory::CreateIdFilterAdapter(id_filter);
      dynamic_search_parameters.sel = id_filter_adapter.get();
    }

    VLOG(VERBOSE_DEBUG) << "range: " << range << ", limit: " << limit
                        << ", nprobe: " << dynamic_search_parameters.nprobe;

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

    // Note that the parameters pass to faiss::IndexPretransform::range_search will be transparently
    // passed to the underlying index
    // (here the params will be passed to tenann::IndexIvfPq::range_search).
    faiss::RangeSearchResult results(ANN_SEARCHER_QUERY_COUNT);
    faiss_index->range_search(ANN_SEARCHER_QUERY_COUNT,
                              reinterpret_cast<const float*>(query_vector.data), radius, &results,
                              &dynamic_search_parameters);

    // number of results returned by index search
    int64_t num_results = results.lims[1];
    // number of results to preserve
    auto num_preserve_results = limit < 0 ? num_results : std::min(num_results, limit);
    result_ids->resize(num_preserve_results);
    result_distances->resize(num_preserve_results);

    const auto* result_id_data = results.labels;
    const auto* result_distance_data = results.distances;

    std::vector<int64_t> indices(num_preserve_results);
    std::iota(indices.begin(), indices.end(), 0);

    auto distance_less = [result_id_data, result_distance_data](int64_t left, int64_t right) {
      if (result_distance_data[left] < result_distance_data[right])
        return true;
      else if (result_distance_data[left] > result_distance_data[right])
        return false;
      else
        return result_id_data[left] < result_id_data[right];
    };
    std::priority_queue<int64_t, std::vector<int64_t>, decltype(distance_less)> heap(distance_less);

    // insert indices to the heap and only preserve top-n results,
    // where n = num_preserve_results
    for (int64_t i = 0; i < num_results; i++) {
      heap.push(i);
      if (heap.size() > num_preserve_results) heap.pop();
    }

    // recording the sorted indices
    indices.resize(num_preserve_results);
    int64_t j = num_preserve_results - 1;
    while (j >= 0) {
      auto idx = heap.top();
      heap.pop();
      indices[j] = idx;
      j -= 1;
    }

    // fetch results by the sorted indices
    for (int64_t i = 0; i < num_preserve_results; i++) {
      auto idx = indices[i];
      (*result_ids)[i] = result_id_data[idx];
      (*result_distances)[i] = result_distance_data[idx];
    }

    if (common_params_.metric_type == MetricType::kCosineSimilarity) {
      auto distances = reinterpret_cast<float*>(result_distances->data());
      L2DistanceToCosineSimilarity(distances, distances, result_distances->size());
    }
  }
  CATCH_FAISS_ERROR
}

void FaissIvfPqAnnSearcher::OnSearchParamItemChange(const std::string& key, const json& value) {
  try {
    if (key == FaissIvfPqSearchParams::nprobe_key) {
      search_params_.nprobe = value.get<FaissIvfPqSearchParams::nprobe_type>();
      return;
    }

    if (key == FaissIvfPqSearchParams::max_codes_key) {
      search_params_.max_codes = value.get<FaissIvfPqSearchParams::max_codes_type>();
      return;
    }

    if (key == FaissIvfPqSearchParams::scan_table_threshold_key) {
      search_params_.scan_table_threshold =
          value.get<FaissIvfPqSearchParams::scan_table_threshold_type>();
      return;
    }

    if (key == FaissIvfPqSearchParams::polysemous_ht_key) {
      search_params_.polysemous_ht = value.get<FaissIvfPqSearchParams::polysemous_ht_type>();
      return;
    }

    if (key == FaissIvfPqSearchParams::range_search_confidence_key) {
      search_params_.range_search_confidence =
          value.get<FaissIvfPqSearchParams::range_search_confidence_type>();
      return;
    }
  } catch (json::exception& e) {
    T_LOG(ERROR) << "failed to get search parameter from json: " << e.what();
  }

  T_LOG(ERROR) << "unsupport search parameter: " << key;
};

void FaissIvfPqAnnSearcher::FaissIvfPqAnnSearcher::OnSearchParamsChange(const json& value) {
  for (auto it = value.begin(); it != value.end(); ++it) {
    OnSearchParamItemChange(it.key(), it.value());
  }
}

}  // namespace tenann