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

#include "stddef.h"

namespace tenann {

/**
 * @brief Convert l2 (euclidean square) distance to cosine simimarity.
 * It only works if both the database and query vectors are nomalized.
 *
 * @param src Source
 * @param dst Destination
 * @param k   Size of inputs
 */
inline void L2DistanceToCosineSimilarity(const float* src, float* dst, size_t k) {
  for (size_t i = 0; i < k; i++) {
    dst[i] = 1 - src[i] / 2;
  }
}

}  // namespace tenann
