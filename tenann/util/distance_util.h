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

#include <stddef.h>

#include <queue>

#include "tenann/common/error.h"
#include "tenann/common/macros.h"

namespace tenann {

/**
 * @brief Convert l2 (euclidean square) distance to cosine simimarity.
 * It only works if both the database and query vectors are normalized.
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

/**
 * @brief Used for range search. Convert a cosine similarity threshold to l2 distane limit.
 * It only works if both the database and query vectors are normalized.
 *
 * @param threshold Threshold for range search base on cosine similarity
 * @return float
 */

inline float CosineSimilarityThresholdToL2Distance(float threshold) {
  if (threshold < -1 || threshold > 1) {
    throw Error(__FILE__, __LINE__,
                "the give cosine similarity threshold must be in range [-1, 1]");
  }
  return (1 - threshold) * 2;
}

namespace detail {
/// to sort pairs of (id, distance) from nearest to fathest or the reverse
struct NodeDistCloser {
  float d;
  int id;
  NodeDistCloser(float d, int id) : d(d), id(id) {}
  bool operator<(const NodeDistCloser& obj1) const { return d < obj1.d; }
};

struct NodeDistFarther {
  float d;
  int id;
  NodeDistFarther(float d, int id) : d(d), id(id) {}
  bool operator<(const NodeDistFarther& obj1) const { return d > obj1.d; }
};

template <bool ascending = true>
inline void ReserveTopK(std::vector<int64_t>* ids, std::vector<float>* distances,
                        int64_t k) T_THROW_EXCEPTION {
  if (ids->size() != distances->size()) {
    throw Error(__FILE__, __LINE__, "ids and distances must be of the same size");
  }

  auto get_heap = []() {
    if constexpr (ascending) {
      std::priority_queue<NodeDistCloser> heap;
      return heap;
    } else {
      std::priority_queue<NodeDistFarther> heap;
      return heap;
    }
  };
  auto heap = get_heap();

  for (int i = 0; i < ids->size(); i++) {
    auto id = (*ids)[i];
    auto dis = (*distances)[i];
    heap.emplace(dis, id);
    while (heap.size() > k) {
      heap.pop();
    }
  }

  int i = heap.size();
  ids->resize(i);
  distances->resize(i);
  i -= 1;
  while (i >= 0) {
    auto dis = heap.top().d;
    auto id = heap.top().id;
    heap.pop();

    (*ids)[i] = id;
    (*distances)[i] = dis;
    i -= 1;
  }
}
}  // namespace detail

inline void ReserveTopK(std::vector<int64_t>* ids, std::vector<float>* distances, int64_t k,
                        bool ascending = true) {
  if (ascending) {
    detail::ReserveTopK<true>(ids, distances, k);
  } else {
    detail::ReserveTopK<false>(ids, distances, k);
  }
}

}  // namespace tenann
