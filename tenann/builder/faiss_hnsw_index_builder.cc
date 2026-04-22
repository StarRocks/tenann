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
#include "faiss/IndexIDMap.h"
#include "faiss/index_factory.h"
#include "faiss_hnsw_index_builder.h"
#include "tenann/common/logging.h"
#include "tenann/common/typed_seq_view.h"
#include "tenann/index/index.h"
#include "tenann/index/internal/faiss_index_util.h"
#include "tenann/index/parameter_serde.h"

namespace tenann {

FaissHnswIndexBuilder::FaissHnswIndexBuilder(const IndexMeta& meta)
    : FaissIndexBuilderWithBuffer(meta) {
  FetchParameters(meta, &index_params_);
  FetchParameters(meta, &search_params_);

  auto q = index_params_.quantizer;
  T_CHECK(q == static_cast<int>(ScalarQuantizerType::kFlat) ||
          q == static_cast<int>(ScalarQuantizerType::kSQ4) ||
          q == static_cast<int>(ScalarQuantizerType::kSQ8) ||
          q == static_cast<int>(ScalarQuantizerType::kPQ))
      << "invalid HNSW quantizer value: " << q;

  if (q != static_cast<int>(ScalarQuantizerType::kFlat)) {
    T_CHECK_EQ(common_params_.metric_type, MetricType::kL2Distance)
        << "HNSW quantized variants (SQ/PQ) currently only support L2 metric";
  }

  if (q == static_cast<int>(ScalarQuantizerType::kPQ)) {
    T_CHECK_GE(index_params_.nbits_pq, 4);
    T_CHECK_LE(index_params_.nbits_pq, 16);
    T_CHECK_GT(index_params_.m_pq, 0);
    T_CHECK_EQ(common_params_.dim % index_params_.m_pq, 0)
        << "HNSW+PQ requires dim (" << common_params_.dim
        << ") to be divisible by m_pq (" << index_params_.m_pq << ")";
  }
}

FaissHnswIndexBuilder::~FaissHnswIndexBuilder() {}

size_t FaissHnswIndexBuilder::GetMinTrainRows() const {
  switch (static_cast<ScalarQuantizerType>(index_params_.quantizer)) {
    case ScalarQuantizerType::kPQ:
      // faiss' recommendation: at least 100 training rows per centroid; the
      // PQ codebook has 2^nbits_pq centroids.
      return (static_cast<size_t>(1) << index_params_.nbits_pq) * 100;
    case ScalarQuantizerType::kSQ4:
    case ScalarQuantizerType::kSQ8:
      return 1;
    case ScalarQuantizerType::kFlat:
    default:
      return 0;
  }
}

IndexRef FaissHnswIndexBuilder::InitIndex() {
  try {
    // create faiss index factory string
    auto factory_string =
        faiss_util::GetHnswRepr(common_params_, index_params_, use_custom_row_id_);

    auto metric_type = faiss::METRIC_L2;
    if (common_params_.metric_type == MetricType::kInnerProduct) {
      metric_type = faiss::METRIC_INNER_PRODUCT;
    }

    // create faiss index
    auto index = std::unique_ptr<faiss::Index>(
        faiss::index_factory(common_params_.dim, factory_string.c_str(), metric_type));
    auto [_, __, index_hnsw] = faiss_util::CheckAndUnpackHnswMutable(index.get(), &common_params_);

    // set index parameters
    index_hnsw->hnsw.efConstruction = index_params_.efConstruction;
    // set default search paremeters
    index_hnsw->hnsw.efSearch = search_params_.efSearch;
    index_hnsw->hnsw.check_relative_distance = search_params_.check_relative_distance;

    VLOG(VERBOSE_DEBUG) << "efConstruction: " << index_hnsw->hnsw.efConstruction
                        << ", efSearch: " << index_hnsw->hnsw.efSearch
                        << ", check_relative_distance: " << index_hnsw->hnsw.check_relative_distance;

    // create shared index ref
    return std::make_shared<Index>(index.release(),        //
                                   IndexType::kFaissHnsw,  //
                                   [](void* index) { delete static_cast<faiss::Index*>(index); });
  }
  CATCH_FAISS_ERROR
  CATCH_JSON_ERROR
}

}  // namespace tenann