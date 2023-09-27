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
#include "tenann/common/logging.h"
#include "tenann/common/typed_seq_view.h"
#include "tenann/index/index.h"

namespace tenann {

FaissIvfPqIndexBuilder::~FaissIvfPqIndexBuilder() {}

IndexRef FaissIvfPqIndexBuilder::InitIndex() {
  try {
    std::ostringstream oss;
    oss << "IVF";
    if (index_meta_.index_params().contains("nlist")) {
      oss << index_meta_.index_params()["nlist"].get<int>();
    }
    oss << ",PQ";
    if (index_meta_.index_params().contains("M")) {
      oss << index_meta_.index_params()["M"].get<int>();
    }
    if (index_meta_.index_params().contains("nbits")) {
      oss << "x" << index_meta_.index_params()["nbits"].get<int>();
    }

    auto index = std::unique_ptr<faiss::Index>(
        faiss::index_factory(common_params_.dim, oss.str().c_str(), faiss::METRIC_L2));
    faiss::IndexIVFPQ* index_ivf_pq = nullptr;
    index_ivf_pq = static_cast<faiss::IndexIVFPQ*>(index.get());

    // default search params
    if (index_meta_.search_params()[FAISS_SEARCHER_PARAMS_IVF_NPROBE].is_number_unsigned()) {
      index_ivf_pq->nprobe =
          index_meta_.search_params()[FAISS_SEARCHER_PARAMS_IVF_NPROBE].get<size_t>();
    }

    if (index_meta_.search_params()[FAISS_SEARCHER_PARAMS_IVF_MAX_CODES].is_number_unsigned()) {
      index_ivf_pq->max_codes =
          index_meta_.search_params()[FAISS_SEARCHER_PARAMS_IVF_MAX_CODES].get<size_t>();
    }

    if (index_meta_.search_params()[FAISS_SEARCHER_PARAMS_IVF_PQ_SCAN_TABLE_THRESHOLD]
            .is_number_unsigned()) {
      index_ivf_pq->scan_table_threshold =
          index_meta_.search_params()[FAISS_SEARCHER_PARAMS_IVF_PQ_SCAN_TABLE_THRESHOLD]
              .get<size_t>();
    }

    if (index_meta_.search_params()[FAISS_SEARCHER_PARAMS_IVF_PQ_POLYSEMOUS_HT]
            .is_number_integer()) {
      index_ivf_pq->polysemous_ht =
          index_meta_.search_params()[FAISS_SEARCHER_PARAMS_IVF_PQ_POLYSEMOUS_HT].get<int>();
    }

    return std::make_shared<Index>(index.release(), IndexType::kFaissIvfPq,
                                   [](void* index) { delete static_cast<faiss::Index*>(index); });
  }
  CATCH_FAISS_ERROR
  CATCH_JSON_ERROR
}

}  // namespace tenann