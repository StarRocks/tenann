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
#include "tenann/index/parameter_serde.h"

namespace tenann {
FaissIvfPqIndexBuilder::FaissIvfPqIndexBuilder(const IndexMeta& meta)
    : FaissIndexBuilderWithBuffer(meta) {
  FetchParameters(meta, &index_params_);
  FetchParameters(meta, &search_params_);
}

FaissIvfPqIndexBuilder::~FaissIvfPqIndexBuilder() = default;

IndexRef FaissIvfPqIndexBuilder::InitIndex() {
  try {
    auto factory_string = faiss_util::GetIvfPqRepr(common_params_, index_params_);
    auto index = std::unique_ptr<faiss::Index>(
        faiss::index_factory(common_params_.dim, factory_string.c_str(), faiss::METRIC_L2));

    faiss::IndexIVFPQ* index_ivf_pq = static_cast<faiss::IndexIVFPQ*>(index.get());

    // default search params
    index_ivf_pq->nprobe = search_params_.nprobe;
    index_ivf_pq->max_codes = search_params_.max_codes;
    index_ivf_pq->scan_table_threshold = search_params_.scan_table_threshold;
    index_ivf_pq->polysemous_ht = search_params_.polysemous_ht;

    return std::make_shared<Index>(index.release(),         //
                                   IndexType::kFaissIvfPq,  //
                                   [](void* index) { delete static_cast<faiss::Index*>(index); });
  }
  CATCH_FAISS_ERROR
  CATCH_JSON_ERROR
}

}  // namespace tenann