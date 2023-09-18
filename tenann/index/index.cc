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

#include "tenann/index/index.h"

#include "faiss/Index.h"
#include "faiss/IndexHNSW.h"
#include "faiss/IndexIVFPQ.h"
#include "tenann/common/logging.h"

namespace tenann {

Index::~Index() noexcept {
  if (index_raw_ != nullptr) {
    deleter_(index_raw_);
  }
}

Index::Index(Index&& rhs) noexcept { std::swap(*this, rhs); }

Index::Index(void* index, IndexType index_type, const std::function<void(void*)>& deleter) noexcept
    : index_raw_(index), index_type_(index_type), deleter_(deleter) {}

Index& Index::operator=(Index&& rhs) noexcept {
  std::swap(*this, rhs);
  return *this;
}

void Index::SetIndexRaw(void* index) { index_raw_ = index; }

void Index::SetIndexType(IndexType index_type) { index_type_ = index_type; }

void* Index::index_raw() const { return index_raw_; }

IndexType Index::index_type() const { return index_type_; }

// @TODO(petri): implement it with a seperate class
size_t Index::EstimateMemoryUsage() {
  if (index_type_ == IndexType::kFaissHnsw) {
    auto* faiss_index = static_cast<faiss::Index*>(index_raw_);
    auto* index_hnsw = dynamic_cast<faiss::IndexHNSW*>(faiss_index);
    // TODO: always is nullptr now
    if (index_hnsw == nullptr) {
      T_LOG(WARNING)
          << "estimating memory usage for unsupported index types would always get result 1";
      return 1;
    }

    auto& hnsw = index_hnsw->hnsw;
    auto mem_usage = sizeof(index_hnsw);

    // graph structure
    mem_usage += hnsw.assign_probas.capacity() * sizeof(double) +
                 hnsw.cum_nneighbor_per_level.capacity() * sizeof(int) +
                 hnsw.levels.capacity() * sizeof(int) + hnsw.offsets.capacity() * sizeof(size_t) +
                 hnsw.neighbors.capacity() * sizeof(faiss::HNSW::storage_idx_t);

    // vectors
    mem_usage += index_hnsw->storage->ntotal * index_hnsw->storage->d * sizeof(float);

    return mem_usage;
  } else if (index_type_ == IndexType::kFaissIvfPq) {
    auto* faiss_index = static_cast<faiss::Index*>(index_raw_);
    auto* index_ivf_pq = dynamic_cast<faiss::IndexIVFPQ*>(faiss_index);
    if (index_ivf_pq == nullptr) {
      T_LOG(WARNING)
          << "estimating memory usage for unsupported index types would always get result 1";
      return 1;
    }

    auto mem_usage = sizeof(index_ivf_pq);

    // TODO: add mem_usage calc
    return mem_usage;
  } else {
    T_LOG(ERROR) << "not implemented yet";
  }
}

}  // namespace tenann