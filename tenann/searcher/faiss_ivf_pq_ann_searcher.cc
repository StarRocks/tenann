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

#include "faiss_ivf_pq_ann_searcher.h"
#include "tenann/common/logging.h"
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

  faiss_index->search(ANN_SEARCHER_QUERY_COUNT, reinterpret_cast<const float*>(query_vector.data),
                      k, reinterpret_cast<float*>(result_distances), result_ids,
                      search_parameters_.get());

  auto distances = reinterpret_cast<float*>(result_distances);
  if (common_params_.metric_type == MetricType::kCosineSimilarity) {
    L2DistanceToCosineSimilarity(distances, distances, k);
  }
}

void FaissIvfPqAnnSearcher::SearchParamItemChangeHook(const std::string& key, const json& value) {
  if (!search_parameters_) {
    search_parameters_ = std::make_unique<faiss::IVFPQSearchParameters>();
  }

  if (key == FAISS_SEARCHER_PARAMS_IVF_NPROBE) {
    value.is_number_unsigned() && (search_parameters_->nprobe = value.get<size_t>());
  } else if (key == FAISS_SEARCHER_PARAMS_IVF_MAX_CODES) {
    value.is_number_unsigned() && (search_parameters_->max_codes = value.get<size_t>());
  } else if (key == FAISS_SEARCHER_PARAMS_IVF_PQ_SCAN_TABLE_THRESHOLD) {
    value.is_number_unsigned() && (search_parameters_->scan_table_threshold = value.get<size_t>());
  } else if (key == FAISS_SEARCHER_PARAMS_IVF_PQ_POLYSEMOUS_HT) {
    value.is_number_integer() && (search_parameters_->polysemous_ht = value.get<int>());
  } else {
    T_LOG(ERROR) << "Unsupport search parameter: " << key;
  }
};

void FaissIvfPqAnnSearcher::FaissIvfPqAnnSearcher::SearchParamsChangeHook(const json& value) {
  if (search_parameters_) {
    search_parameters_.reset();
  }

  for (auto it = value.begin(); it != value.end(); ++it) {
    SearchParamItemChangeHook(it.key(), it.value());
  }
}

}  // namespace tenann