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

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "tenann/common/seq_view.h"
#include "tenann/error/logging.h"

namespace tenann {

namespace util {

inline float EuclideanDistance(const float* v1, const float* v2, size_t dim) {
  float sum = 0.0;
  for (size_t i = 0; i < dim; i++) {
    float diff = *(v2 + i) - *(v1 + i);
    sum += diff * diff;
  }

  // return std::sqrt(sum); faiss not sqrt
  return sum;
}

inline float CosineSimilarity(const float* v1, const float* v2, size_t dim) {
  float dot_product = 0.0f;
  float norm_v1 = 0.0f;
  float norm_v2 = 0.0f;

  for (size_t i = 0; i < dim; ++i) {
    dot_product += v1[i] * v2[i];
    norm_v1 += v1[i] * v1[i];
    norm_v2 += v2[i] * v2[i];
  }

  if (norm_v1 == 0.0f || norm_v2 == 0.0f) {
    return 0.0f;
  }

  return dot_product / (sqrtf(norm_v1) * sqrtf(norm_v2));
}

// @TODO: implement it
inline std::vector<int64_t> BruteForceAnn(const SeqView& base_col, const SeqView& query_col,
                                          const uint8_t* null_flags, const int64_t* rowids) {
  T_CHECK(null_flags != nullptr && rowids == nullptr);
}

}  // namespace util
}  // namespace tenann`