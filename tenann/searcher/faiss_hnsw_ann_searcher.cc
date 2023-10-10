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

#include "faiss/IndexHNSW.h"
#include "faiss/IndexIDMap.h"
#include "faiss/impl/HNSW.h"
#include "tenann/common/logging.h"
#include "tenann/index/internal/faiss_index_util.h"
#include "tenann/store/index_meta.h"
#include "tenann/util/distance_util.h"

namespace tenann {

FaissHnswAnnSearcher::FaissHnswAnnSearcher(const IndexMeta& meta) : AnnSearcher(meta) {}

FaissHnswAnnSearcher::~FaissHnswAnnSearcher() = default;

void FaissHnswAnnSearcher::AnnSearch(PrimitiveSeqView query_vector, int k, int64_t* result_id) {
  std::vector<float> distances(k);
  AnnSearch(query_vector, k, result_id, reinterpret_cast<uint8_t*>(distances.data()));
}

void FaissHnswAnnSearcher::AnnSearch(PrimitiveSeqView query_vector, int k, int64_t* result_ids,
                                     uint8_t* result_distances) {
  T_CHECK_NOTNULL(index_ref_);

  T_CHECK_EQ(index_ref_->index_type(), IndexType::kFaissHnsw);
  T_CHECK_EQ(query_vector.elem_type, PrimitiveType::kFloatType);

  auto faiss_index = static_cast<faiss::Index*>(index_ref_->index_raw());
  auto [hnsw_index, idmap_index] = faiss_util::UnpackHnsw(faiss_index, common_params_);

  faiss::SearchParametersHNSW faiss_search_parameters;
  faiss_search_parameters.efSearch = search_params_.efSearch;
  faiss_search_parameters.check_relative_distance = search_params_.check_relative_distance;

  hnsw_index->search(ANN_SEARCHER_QUERY_COUNT, reinterpret_cast<const float*>(query_vector.data), k,
                     reinterpret_cast<float*>(result_distances), result_ids,
                     &faiss_search_parameters);

  if (idmap_index != nullptr) {
    int64_t* li = result_ids;
    for (int64_t i = 0; i < ANN_SEARCHER_QUERY_COUNT * k; i++) {
      li[i] = li[i] < 0 ? li[i] : idmap_index->id_map[li[i]];
    }
  }

  auto distances = reinterpret_cast<float*>(result_distances);
  if (common_params_.metric_type == MetricType::kCosineSimilarity) {
    L2DistanceToCosineSimilarity(distances, distances, k);
  }
}

void FaissHnswAnnSearcher::SearchParamItemChangeHook(const std::string& key, const json& value) {
  if (key == FaissHnswSearchParams::efSearch_key) {
    value.is_number_integer() && (search_params_.efSearch = value.get<int>());
    T_LOG(INFO) << "here:: " << key;
  } else if (key == FaissHnswSearchParams::check_relative_distance_key) {
    value.is_boolean() && (search_params_.check_relative_distance = value.get<bool>());
    T_LOG(INFO) << "here:: " << key;
  } else {
    T_LOG(ERROR) << "Unsupport search parameter: " << key;
  }
};

void FaissHnswAnnSearcher::SearchParamsChangeHook(const json& value) {
  for (auto it = value.begin(); it != value.end(); ++it) {
    SearchParamItemChangeHook(it.key(), it.value());
  }
}
}  // namespace tenann