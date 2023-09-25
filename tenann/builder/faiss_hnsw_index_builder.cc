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

namespace tenann {

FaissHnswIndexBuilder::~FaissHnswIndexBuilder() {}

IndexRef FaissHnswIndexBuilder::InitIndex() {
  // TODO: add "M", "efConstruction", "efSearch" limit check
  try {
    std::ostringstream oss;
    if (use_custom_row_id_) {
      oss << "IDMap,";
    }

    oss << "HNSW";
    if (index_meta_.index_params().contains("M")) {
      oss << index_meta_.index_params()["M"].get<int>();
    }

    auto index = std::unique_ptr<faiss::Index>(
        faiss::index_factory(dim_, oss.str().c_str(), faiss::METRIC_L2));
    faiss::IndexHNSW* index_hnsw = nullptr;
    if (use_custom_row_id_) {
      index_hnsw =
          static_cast<faiss::IndexHNSW*>(static_cast<faiss::IndexIDMap*>(index.get())->index);
    } else {
      index_hnsw = static_cast<faiss::IndexHNSW*>(index.get());
    }

    if (index_meta_.index_params()["efConstruction"].is_number_integer()) {
      index_hnsw->hnsw.efConstruction = index_meta_.index_params()["efConstruction"].get<int>();
    }

    // default search params
    if (index_meta_.search_params()[FAISS_SEARCHER_PARAMS_HNSW_EF_SEARCH].is_number_integer()) {
      index_hnsw->hnsw.efSearch =
          index_meta_.search_params()[FAISS_SEARCHER_PARAMS_HNSW_EF_SEARCH].get<int>();
    }

    if (index_meta_.search_params()[FAISS_SEARCHER_PARAMS_HNSW_CHECK_RELATIVE_DISTANCE]
            .is_boolean()) {
      index_hnsw->hnsw.check_relative_distance =
          index_meta_.search_params()[FAISS_SEARCHER_PARAMS_HNSW_CHECK_RELATIVE_DISTANCE]
              .get<bool>();
    }

    return std::make_shared<Index>(index.release(),        //
                                   IndexType::kFaissHnsw,  //
                                   [](void* index) { delete static_cast<faiss::Index*>(index); });
  }
  CATCH_FAISS_ERROR
  CATCH_JSON_ERROR
}

}  // namespace tenann