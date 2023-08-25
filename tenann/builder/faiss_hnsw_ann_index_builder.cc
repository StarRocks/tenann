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

#include <faiss/IndexHNSW.h>

#include <faiss/index_factory.h>

#include "faiss_hnsw_ann_index_builder.h"

namespace tenann {

// del IndexBuilder()
FaissHnswAnnIndexBuilder::FaissHnswAnnIndexBuilder(const IndexMeta& meta) {
  index_meta_ = meta;
}

void FaissHnswAnnIndexBuilder::BuildWithPrimaryKeyImpl(const std::vector<SeqView>& input_columns,
                                                       int primary_key_column_index) {
}

void FaissHnswAnnIndexBuilder::BuildImpl(const std::vector<SeqView>& input_columns) {

  int d = index_meta_.common_params()["dim"].get<int>();
  /// std::string index_type = "HNSW" + std::to_string(index_meta_.meta_json_["M"].get<int>());
  constexpr const char* index_type = "HNSW64";  /// TODO: setting M in index_meta
  /// meta.common_params()["metric_type"] = MetricType::kL2Distance; TODO: get metric_type from index_meta
  //std::unique_ptr<faiss::Index> index = std::unique_ptr<faiss::Index>(faiss::index_factory(d, index_type, faiss::METRIC_L2));
  auto index = std::unique_ptr<faiss::IndexHNSW>(dynamic_cast<faiss::IndexHNSW*>(faiss::index_factory(d, index_type, faiss::METRIC_L2)));
  index->hnsw.efConstruction = index_meta_.index_params()["efConstruction"].get<int>();
  index->hnsw.efSearch = index_meta_.search_params()["efSearch"].get<int>();
  
  auto array_seq_view = input_columns[0].seq_view.array_seq_view;
  index->add(array_seq_view.size, reinterpret_cast<float*>(array_seq_view.data));

  index_ref_ = std::make_shared<Index>(index.release(), IndexType::kFaissHnsw, [](void* index) {delete static_cast<faiss::IndexHNSW*>(index);});
}

}  // namespace tenann