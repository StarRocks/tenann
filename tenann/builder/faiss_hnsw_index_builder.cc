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

#include "tenann/builder/faiss_hnsw_index_builder.h"

#include <sstream>

#include "faiss/IndexHNSW.h"
#include "faiss/index_factory.h"
#include "tenann/common/logging.h"

namespace tenann {

void FaissHnswIndexBuilder::BuildWithPrimaryKeyImpl(const std::vector<SeqView>& input_columns,
                                                    int primary_key_column_index) {}

void FaissHnswIndexBuilder::BuildImpl(const std::vector<SeqView>& input_columns) {
  T_CHECK(input_columns.size() == 1);
  // @TODO(petri): add support for VlArraySeqView
  T_CHECK(input_columns[0].seq_view_type == SeqViewType::kArraySeqView);
  T_CHECK(input_columns[0].seq_view.array_seq_view.elem_type == PrimitiveType::kFloatType);

  // @TODO(petri): provide a unified macro to report parameter errors
  T_LOG_IF(ERROR, !index_meta_.common_params().contains("dim"))
      << "required common parameter `dim` is not set in index meta";
  T_LOG_IF(ERROR, !index_meta_.common_params().contains("metric_type"))
      << "required common parameter `metric_type` is not set in index meta";

  // @TODO(petri): catch errors thrown from both faiss and the json library
  std::ostringstream oss;
  oss << "HNSW";
  if (index_meta_.index_params().contains("M")) {
    oss << index_meta_.index_params()["M"].get<int>();
  }

  T_LOG_IF(ERROR, index_meta_.common_params()["metric_type"].get<int>() != MetricType::kL2Distance)
      << "only l2 distance is supported now";

  auto dim = index_meta_.common_params()["dim"].get<int>();
  auto index = std::unique_ptr<faiss::IndexHNSW>(static_cast<faiss::IndexHNSW*>(
      faiss::index_factory(dim, oss.str().c_str(), faiss::METRIC_L2)));

  if (index_meta_.index_params().contains("efConstruction")) {
    index->hnsw.efConstruction = index_meta_.index_params()["efConstruction"].get<int>();
  }

  if (index_meta_.search_params().contains("efSearch")) {
    index->hnsw.efSearch = index_meta_.search_params()["efSearch"].get<int>();
  }

  auto array_seq_view = input_columns[0].seq_view.array_seq_view;
  index->add(array_seq_view.size, reinterpret_cast<float*>(array_seq_view.data));

  index_ref_ =
      std::make_shared<Index>(index.release(),        //
                              IndexType::kFaissHnsw,  //
                              [](void* index) { delete static_cast<faiss::IndexHNSW*>(index); });
}

}  // namespace tenann