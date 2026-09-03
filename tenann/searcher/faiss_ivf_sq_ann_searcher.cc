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

#include "tenann/searcher/faiss_ivf_sq_ann_searcher.h"

#include <algorithm>
#include <queue>

#include "faiss/Index.h"
#include "faiss/IndexIVF.h"
#include "faiss/impl/AuxIndexStructures.h"
#include "tenann/common/logging.h"
#include "tenann/index/internal/faiss_index_util.h"
#include "tenann/index/internal/metric_util.h"
#include "tenann/index/parameter_serde.h"
#include "tenann/searcher/internal/id_filter_adapter.h"

namespace tenann {

FaissIvfSqAnnSearcher::FaissIvfSqAnnSearcher(const IndexMeta& meta) : AnnSearcher(meta) {
  FetchParameters(meta, &search_params_);
}

FaissIvfSqAnnSearcher::~FaissIvfSqAnnSearcher() = default;

void FaissIvfSqAnnSearcher::AnnSearch(PrimitiveSeqView query_vector, int64_t k, int64_t* result_ids,
                                      const IdFilter* id_filter) {
  std::vector<float> distances(k);
  AnnSearch(query_vector, k, result_ids, reinterpret_cast<uint8_t*>(distances.data()), id_filter);
}

void FaissIvfSqAnnSearcher::AnnSearch(PrimitiveSeqView query_vector, int64_t k, int64_t* result_ids,
                                      uint8_t* result_distances, const IdFilter* id_filter) {
  try {
    T_CHECK_NOTNULL(index_ref_);
    T_CHECK_EQ(index_ref_->index_type(), IndexType::kFaissIvfSq);
    T_CHECK_EQ(query_vector.elem_type, PrimitiveType::kFloatType);

    auto faiss_index = static_cast<faiss::Index*>(index_ref_->index_raw());
    faiss::SearchParametersIVF faiss_search_parameters;
    faiss_search_parameters.nprobe = search_params_.nprobe;
    faiss_search_parameters.max_codes = search_params_.max_codes;

    std::shared_ptr<IdFilterAdapter> id_filter_adapter;
    if (id_filter != nullptr) {
      id_filter_adapter = IdFilterAdapterFactory::CreateIdFilterAdapter(id_filter);
      faiss_search_parameters.sel = id_filter_adapter.get();
    }

    VLOG(VERBOSE_DEBUG) << "nprobe: " << faiss_search_parameters.nprobe;
    std::vector<float> query_scratch;
    const float* query = PrepareCosineQuery(reinterpret_cast<const float*>(query_vector.data),
                                            query_vector.size, &query_scratch);
    faiss_index->search(ANN_SEARCHER_QUERY_COUNT, query, k,
                        reinterpret_cast<float*>(result_distances), result_ids,
                        &faiss_search_parameters);
    FinalizeScores(result_ids, reinterpret_cast<float*>(result_distances), k, physical_metric_);
  }
  CATCH_FAISS_ERROR
}

void FaissIvfSqAnnSearcher::RangeSearch(PrimitiveSeqView query_vector, float range, int64_t limit,
                                        ResultOrder result_order, std::vector<int64_t>* result_ids,
                                        std::vector<float>* result_distances,
                                        const IdFilter* id_filter) {
  try {
    T_CHECK_NOTNULL(index_ref_);
    T_CHECK_NOTNULL(result_ids);
    T_CHECK_NOTNULL(result_distances);
    T_CHECK_EQ(index_ref_->index_type(), IndexType::kFaissIvfSq);
    T_CHECK_EQ(query_vector.elem_type, PrimitiveType::kFloatType);

    faiss::SearchParametersIVF faiss_search_parameters;
    faiss_search_parameters.nprobe = search_params_.nprobe;
    faiss_search_parameters.max_codes = search_params_.max_codes;
    std::shared_ptr<IdFilterAdapter> id_filter_adapter;
    if (id_filter != nullptr) {
      id_filter_adapter = IdFilterAdapterFactory::CreateIdFilterAdapter(id_filter);
      faiss_search_parameters.sel = id_filter_adapter.get();
    }

    float radius = range;
    if (common_params_.metric_type == MetricType::kCosineSimilarity) {
      radius = PrepareCosineRange(range, physical_metric_);
      T_CHECK(result_order == ResultOrder::kDescending)
          << "only descending order is allowed for cosine similarity range search";
    } else if (common_params_.metric_type == MetricType::kInnerProduct) {
      T_CHECK(result_order == ResultOrder::kDescending)
          << "only descending order is allowed for inner product range search";
    } else if (common_params_.metric_type == MetricType::kL2Distance) {
      T_CHECK(result_order == ResultOrder::kAscending)
          << "only ascending order is allowed for L2 distance range search";
    } else {
      T_CHECK(false)
          << "IVF-SQ range search only supports L2 distance, cosine similarity, and inner product";
    }

    auto faiss_index = static_cast<const faiss::Index*>(index_ref_->index_raw());
    std::vector<float> query_scratch;
    const float* query = PrepareCosineQuery(reinterpret_cast<const float*>(query_vector.data),
                                            query_vector.size, &query_scratch);
    faiss::RangeSearchResult results(ANN_SEARCHER_QUERY_COUNT);
    faiss_index->range_search(ANN_SEARCHER_QUERY_COUNT, query, radius, &results,
                              &faiss_search_parameters);

    const int64_t num_results = results.lims[1];
    const int64_t num_preserved = limit < 0 ? num_results : std::min(num_results, limit);
    result_ids->resize(num_preserved);
    result_distances->resize(num_preserved);
    if (num_preserved == 0) {
      return;
    }

    const bool is_similarity = physical_metric_ == MetricType::kInnerProduct;
    auto better_last = [&results, is_similarity](int64_t left, int64_t right) {
      if (results.distances[left] != results.distances[right]) {
        return is_similarity ? results.distances[left] > results.distances[right]
                             : results.distances[left] < results.distances[right];
      }
      return results.labels[left] < results.labels[right];
    };
    std::priority_queue<int64_t, std::vector<int64_t>, decltype(better_last)> heap(better_last);
    for (int64_t i = 0; i < num_results; ++i) {
      heap.push(i);
      if (static_cast<int64_t>(heap.size()) > num_preserved) {
        heap.pop();
      }
    }

    std::vector<int64_t> indices(num_preserved);
    for (int64_t i = num_preserved; i > 0; --i) {
      indices[i - 1] = heap.top();
      heap.pop();
    }
    for (int64_t i = 0; i < num_preserved; ++i) {
      (*result_ids)[i] = results.labels[indices[i]];
      (*result_distances)[i] = results.distances[indices[i]];
    }

    FinalizeScores(result_ids->data(), result_distances->data(), result_distances->size(),
                   physical_metric_);
  }
  CATCH_FAISS_ERROR
}

void FaissIvfSqAnnSearcher::OnSearchParamItemChange(const std::string& key, const json& value) {
  try {
    if (key == FaissIvfSqSearchParams::nprobe_key) {
      search_params_.nprobe = value.get<FaissIvfSqSearchParams::nprobe_type>();
      search_params_.Validate();
      return;
    }
    if (key == FaissIvfSqSearchParams::max_codes_key) {
      search_params_.max_codes = value.get<FaissIvfSqSearchParams::max_codes_type>();
      search_params_.Validate();
      return;
    }
  } catch (json::exception& e) {
    T_LOG(ERROR) << "failed to get IVF-SQ search parameter from json: " << e.what();
  }
  T_LOG(ERROR) << "unsupported IVF-SQ search parameter: " << key;
}

void FaissIvfSqAnnSearcher::OnSearchParamsChange(const json& value) {
  for (auto it = value.begin(); it != value.end(); ++it) {
    OnSearchParamItemChange(it.key(), it.value());
  }
}

void FaissIvfSqAnnSearcher::OnIndexLoaded() {
  auto* faiss_index = static_cast<faiss::Index*>(index_ref_->index_raw());
  auto [_, ivf_sq] = faiss_util::CheckAndUnpackIvfSq(faiss_index, &common_params_);
  T_CHECK_NOTNULL(ivf_sq->quantizer);
  T_CHECK_EQ(ivf_sq->metric_type, ivf_sq->quantizer->metric_type)
      << "IVF-SQ metric does not match its coarse quantizer metric";
  physical_metric_ = FromFaissMetric(ivf_sq->metric_type);
  ValidateLoadedMetric(static_cast<MetricType>(common_params_.metric_type), physical_metric_);
}

}  // namespace tenann
