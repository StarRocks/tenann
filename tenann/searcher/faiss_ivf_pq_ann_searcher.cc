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

#include "faiss/IndexIVFPQ.h"
#include "faiss_ivf_pq_ann_searcher.h"
#include "tenann/common/logging.h"
#include "tenann/index/parameters.h"
#include "tenann/util/distance_util.h"

namespace tenann {

FaissIvfPqAnnSearcher::FaissIvfPqAnnSearcher(const IndexMeta& meta) : AnnSearcher(meta) {}

FaissIvfPqAnnSearcher::~FaissIvfPqAnnSearcher() = default;

void FaissIvfPqAnnSearcher::AnnSearch(PrimitiveSeqView query_vector, int k, int64_t* result_id) {
  std::vector<float> distances(k);
  AnnSearch(query_vector, k, result_id, reinterpret_cast<uint8_t*>(distances.data()));
}

void FaissIvfPqAnnSearcher::AnnSearch(PrimitiveSeqView query_vector, int k, int64_t* result_ids,
                                      uint8_t* result_distances) {
  T_CHECK_NOTNULL(index_ref_);

  T_CHECK_EQ(index_ref_->index_type(), IndexType::kFaissIvfPq);
  T_CHECK_EQ(query_vector.elem_type, PrimitiveType::kFloatType);

  auto faiss_index = static_cast<faiss::Index*>(index_ref_->index_raw());

  faiss::IVFPQSearchParameters faiss_search_parameters;
  faiss_search_parameters.nprobe = search_params_.nprobe;
  faiss_search_parameters.max_codes = search_params_.max_codes;
  faiss_search_parameters.polysemous_ht = search_params_.polysemous_ht;
  faiss_search_parameters.scan_table_threshold = search_params_.scan_table_threshold;

  faiss_index->search(ANN_SEARCHER_QUERY_COUNT, reinterpret_cast<const float*>(query_vector.data),
                      k, reinterpret_cast<float*>(result_distances), result_ids,
                      &faiss_search_parameters);

  if (common_params_.metric_type == MetricType::kCosineSimilarity) {
    auto distances = reinterpret_cast<float*>(result_distances);
    L2DistanceToCosineSimilarity(distances, distances, k);
  }
}

void FaissIvfPqAnnSearcher::SearchParamItemChangeHook(const std::string& key, const json& value) {
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
  } catch (json::exception& e) {
    T_LOG(ERROR) << "failed to get search parameter from json: " << e.what();
  }

  T_LOG(ERROR) << "unsupport search parameter: " << key;
};

void FaissIvfPqAnnSearcher::FaissIvfPqAnnSearcher::SearchParamsChangeHook(const json& value) {
  for (auto it = value.begin(); it != value.end(); ++it) {
    SearchParamItemChangeHook(it.key(), it.value());
  }
}

}  // namespace tenann