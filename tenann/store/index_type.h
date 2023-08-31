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

#include <cstdint>

#include "tenann/common/json.hpp"

namespace tenann {

enum IndexFamily { kVectorIndex = 0, kTextIndex };

enum IndexType {
  kFaissHnsw = 0,  // 0: faiss hnsw
  kFaissIvfFlat,   // 1: faiss ivf-flat
  kFaissIvfPq      // 2: faiss ivf-pq
};

enum MetricType {
  kL2Distance = 0,    // 0: euclidean l2 distance
  kCosineSimilarity,  // 1: cosine similarity
  kInnerProduct,      // 2: inner product or dot product
  kCosineDistance,    // 3: cosine distance = 1 - cosine similarity
};

}  // namespace tenann