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

  auto distances = reinterpret_cast<float*>(result_distances);
  auto faiss_index = static_cast<faiss::Index*>(index_ref_->index_raw());
  faiss_index->search(1, reinterpret_cast<const float*>(query_vector.data), k,
                      reinterpret_cast<float*>(result_distances), result_ids,
                      search_parameters_.get());

  if (metric_type_ == MetricType::kCosineSimilarity) {
    for (int i = 0; i < k; i++) {
      distances[i] = 1 - distances[i] / 2;
    }
  }
};

void FaissHnswAnnSearcher::SearchParamItemChangeHook(const std::string& key, const json& value) {
  // TODO:
  // 如果外层使用了 IndexIDMapTemplate，则不支持传入查询参数 faiss/IndexIDMap.cpp:82
  // 1、考虑将use_custom_row_id_参数整合进 index_meta 用于指导 IndexHNSW* 的转换方式
  // 2、考虑将 kFaissHnsw 扩展成 kFaissHnsw 和 kFaissIdMapHnsw、用于指导 IndexHNSW* 的转换方式
  // 3、考虑直接修改 HNSW* 的查询参数默认值，绕过 IDMap<Index>对查询参数的检查
  // 4、考虑引入IndexIDMap相关PR：https://github.com/facebookresearch/faiss/issues/2843，对三方库打补丁
  return;
};

void FaissHnswAnnSearcher::SearchParamsChangeHook(const json& value) {
  for (auto it = value.begin(); it != value.end(); ++it) {
    SearchParamItemChangeHook(it.key(), it.value());
  }
}
}  // namespace tenann