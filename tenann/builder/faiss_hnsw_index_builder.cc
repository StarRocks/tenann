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

FaissHnswIndexBuilder::~FaissHnswIndexBuilder() {}

IndexRef FaissHnswIndexBuilder::InitIndex() {
  // TODO: add "M", "efConstruction", "efSearch" limit check
  try {
    // init index/search parameters from meta
    FetchParameters(index_meta_, &index_params_);
    FetchParameters(index_meta_, &search_params_);

    // create faiss index factory string
    auto factory_string =
        faiss_util::GetHnswRepr(common_params_, index_params_, use_custom_row_id_);

    // create faiss index
    auto index = std::unique_ptr<faiss::Index>(
        faiss::index_factory(common_params_.dim, factory_string.c_str(), faiss::METRIC_L2));
    auto [index_hnsw, _] = faiss_util::UnpackHnswMutable(index.get(), common_params_);

    // set index parameters
    index_hnsw->hnsw.efConstruction = index_params_.efConstruction;
    // set default search paremeters
    index_hnsw->hnsw.efSearch = search_params_.efSearch;
    index_hnsw->hnsw.check_relative_distance = search_params_.check_relative_distance;

    // create shared index ref
    return std::make_shared<Index>(index.release(),        //
                                   IndexType::kFaissHnsw,  //
                                   [](void* index) { delete static_cast<faiss::Index*>(index); });
  }
  CATCH_FAISS_ERROR
  CATCH_JSON_ERROR
}

}  // namespace tenann