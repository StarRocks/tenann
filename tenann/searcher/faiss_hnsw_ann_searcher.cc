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

#include "faiss/IndexIDMap.h"
#include "tenann/common/logging.h"

namespace tenann {
FaissHnswAnnSearcher::FaissHnswAnnSearcher(const IndexMeta& meta) : AnnSearcher(meta) {
  GET_META_OR_DEFAULT(index_meta_, common, "is_vector_normed", bool, is_vector_normed_, false);

  int metric_type_value;
  CHECK_AND_GET_META(index_meta_, common, "metric_type", int, metric_type_value);
  metric_type_ = static_cast<MetricType>(metric_type_value);
}

void FaissHnswAnnSearcher::AnnSearch(PrimitiveSeqView query_vector, int k, int64_t* result_id) {
  std::vector<float> distances(k);
  AnnSearch(query_vector, k, result_id, reinterpret_cast<uint8_t*>(distances.data()));
}

void FaissHnswAnnSearcher::AnnSearch(PrimitiveSeqView query_vector, int k, int64_t* result_ids,
                                     uint8_t* result_distances) {
  T_CHECK_NOTNULL(index_ref_);

  T_CHECK_EQ(index_ref_->index_type(), IndexType::kFaissHnsw);
  T_CHECK_EQ(query_vector.elem_type, PrimitiveType::kFloatType);

  bool use_custom_row_id = false;
  auto faiss_index = static_cast<faiss::Index*>(index_ref_->index_raw());
  faiss::IndexHNSW* hnsw_index = dynamic_cast<faiss::IndexHNSW*>(faiss_index);
  faiss::IndexIDMap* idmap_index = nullptr;

  if (hnsw_index == nullptr) {
    if (idmap_index = dynamic_cast<faiss::IndexIDMap*>(faiss_index)) {
      hnsw_index = dynamic_cast<faiss::IndexHNSW*>(idmap_index->index);
      use_custom_row_id = true;
    } else {
      throw Error(__FILE__, __LINE__, "IndexIDMap parsing error.");
    }

    if (hnsw_index == nullptr) {
      throw Error(__FILE__, __LINE__, "IndexIDMap<IndexHNSW> parsing error.");
    }
  }

  hnsw_index->search(ANN_SEARCHER_QUERY_COUNT, reinterpret_cast<const float*>(query_vector.data), k,
                     reinterpret_cast<float*>(result_distances), result_ids,
                     search_parameters_.get());

  if (use_custom_row_id && idmap_index != nullptr) {
    int64_t* li = result_ids;
    for (int64_t i = 0; i < ANN_SEARCHER_QUERY_COUNT * k; i++) {
      li[i] = li[i] < 0 ? li[i] : idmap_index->id_map[li[i]];
    }
  }

  auto distances = reinterpret_cast<float*>(result_distances);
  if (metric_type_ == MetricType::kCosineSimilarity) {
    for (int i = 0; i < k; i++) {
      distances[i] = 1 - distances[i] / 2;
    }
  }
}

void FaissHnswAnnSearcher::SearchParamItemChangeHook(const std::string& key, const json& value) {
  if (!search_parameters_) {
    search_parameters_ = std::make_unique<faiss::SearchParametersHNSW>();
  }

  if (key == FAISS_SEARCHER_PARAMS_HNSW_EF_SEARCH) {
    value.is_number_integer() && (search_parameters_->efSearch = value.get<int>());
    T_LOG(INFO) << "here:: " << key;

  } else if (key == FAISS_SEARCHER_PARAMS_HNSW_CHECK_RELATIVE_DISTANCE) {
    value.is_boolean() && (search_parameters_->check_relative_distance = value.get<bool>());
    T_LOG(INFO) << "here:: " << key;

  } else {
    T_LOG(ERROR) << "Unsupport search parameter: " << key;
  }
};

void FaissHnswAnnSearcher::SearchParamsChangeHook(const json& value) {
  if (search_parameters_) {
    search_parameters_.reset();
  }

  for (auto it = value.begin(); it != value.end(); ++it) {
    SearchParamItemChangeHook(it.key(), it.value());
  }
}
}  // namespace tenann