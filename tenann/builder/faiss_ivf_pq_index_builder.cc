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

#include "tenann/builder/faiss_ivf_pq_index_builder.h"

#include <sstream>

#include "faiss/IndexIVFPQ.h"
#include "faiss/index_factory.h"
#include "faiss_ivf_pq_index_builder.h"
#include "tenann/common/logging.h"
#include "tenann/common/typed_seq_view.h"
#include "tenann/index/index.h"
#include "tenann/index/internal/faiss_index_util.h"
#include "tenann/index/internal/index_ivfpq.h"
#include "tenann/index/parameter_serde.h"

namespace tenann {
FaissIvfPqIndexBuilder::FaissIvfPqIndexBuilder(const IndexMeta& meta)
    : FaissIndexBuilderWithBuffer(meta) {
  FetchParameters(meta, &index_params_);
  FetchParameters(meta, &search_params_);
  T_CHECK(common_params_.metric_type == MetricType::kL2Distance ||
          common_params_.metric_type == MetricType::kCosineSimilarity ||
          common_params_.metric_type == MetricType::kInnerProduct)
      << "got unsupported metric, l2_distance, kCosineSimilarity and kInnerProduct are supported "
         "for IVF-PQ";
}

FaissIvfPqIndexBuilder::~FaissIvfPqIndexBuilder() = default;

IndexRef FaissIvfPqIndexBuilder::InitIndex() {
  try {
    // create faiss index
    auto metric_type = faiss::METRIC_L2;
    if (common_params_.metric_type == MetricType::kInnerProduct) {
      metric_type = faiss::METRIC_INNER_PRODUCT;
    }

    // use bruteforce coarse quantizer by default
    auto quantizer = std::make_unique<faiss::IndexFlat>(common_params_.dim, metric_type);
    auto index_ivfpq =
        std::make_unique<IndexIvfPq>(quantizer.release(), quantizer->d, index_params_.nlist,
                                     index_params_.M, index_params_.nbits, metric_type);
    index_ivfpq->own_fields = true;

    // default search params
    index_ivfpq->nprobe = search_params_.nprobe;
    index_ivfpq->max_codes = search_params_.max_codes;
    index_ivfpq->scan_table_threshold = search_params_.scan_table_threshold;
    index_ivfpq->polysemous_ht = search_params_.polysemous_ht;
    index_ivfpq->range_search_confidence = search_params_.range_search_confidence;
    // Based on this function: fix_ivf_fields(IndexIVF* index_ivf)
    // Extract the key steps.
    index_ivfpq->quantizer_trains_alone = 0;
    index_ivfpq->cp.spherical = metric_type == faiss::METRIC_INNER_PRODUCT;

    VLOG(VERBOSE_DEBUG) << "nlist: " << index_ivfpq->invlists->nlist << ", M: " << index_ivfpq->pq.M
                        << ", nbits: " << index_ivfpq->pq.nbits;

    if (common_params_.metric_type == MetricType::kCosineSimilarity ||
        common_params_.metric_type == MetricType::kInnerProduct ||
        common_params_.is_vector_normed) {
      auto index_pt = std::make_unique<faiss::IndexPreTransform>(index_ivfpq.release());
      index_pt->own_fields = true;
      auto vector_transform =
          std::make_unique<faiss::NormalizationTransform>(common_params_.dim, 2.0);
      index_pt->prepend_transform(vector_transform.release());
      return std::make_shared<Index>(
          index_pt.release(),      //
          IndexType::kFaissIvfPq,  //
          [](void* index) { delete static_cast<faiss::IndexPreTransform*>(index); });
    }

    return std::make_shared<Index>(index_ivfpq.release(),   //
                                   IndexType::kFaissIvfPq,  //
                                   [](void* index) { delete static_cast<IndexIvfPq*>(index); });
  }
  CATCH_FAISS_ERROR
  CATCH_JSON_ERROR
}

}  // namespace tenann