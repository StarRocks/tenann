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

#pragma once

#include <string>

#include "fmt/format.h"
#include "tenann/index/parameter_serde.h"
#include "tenann/store/index_meta.h"

namespace tenann {

inline std::string IndexStr(const IndexMeta& meta) {
  switch (meta.index_type()) {
    case IndexType::kFaissHnsw: {
      FaissHnswIndexParams params;
      FetchParameters(meta, &params);
      return fmt::format("hnsw{}_efConstruction{}", params.M, params.efConstruction);
    }
    case IndexType::kFaissIvfPq: {
      FaissIvfPqIndexParams params;
      FetchParameters(meta, &params);
      return fmt::format("ivf{}pq{}x{}", params.nlist, params.nbits, params.M);
    }
  }

  return "unknown index";
}

}  // namespace tenann