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
#include "faiss/IndexIDMap.h"
#include "faiss/IndexIVFPQ.h"
#include "tenann/common/logging.h"
#include "tenann/index/internal/faiss_index_util.h"

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
  size_t mem_usage = 0;
  // IndexType::kFaissHnsw
  if (index_type_ == IndexType::kFaissHnsw) {
    auto* faiss_index = static_cast<faiss::Index*>(index_raw_);
    auto [index_id_map, transform, index_hnsw] = faiss_util::UnpackHnsw(faiss_index);

    if (index_hnsw == nullptr) {
      T_LOG(WARNING)
          << "estimating memory usage for unsupported index types would always get result 1";
      return 1;
    }

    // IndexIDMap
    if (index_id_map != nullptr) {
      mem_usage += sizeof(*index_id_map);
      mem_usage += index_id_map->id_map.capacity() * sizeof(faiss::Index::idx_t);
    }

    // IndexPreTransform
    if (transform != nullptr) {
      mem_usage += sizeof(*transform);
      // IndexPreTransform.VectorTransform
      mem_usage += transform->chain.capacity() * sizeof(faiss::VectorTransform*);
      for (auto chain_ptr : transform->chain) {
        mem_usage += sizeof(*chain_ptr);
      }
    }

    auto& hnsw = index_hnsw->hnsw;
    // graph structure
    mem_usage += hnsw.assign_probas.capacity() * sizeof(double) +
                 hnsw.cum_nneighbor_per_level.capacity() * sizeof(int) +
                 hnsw.levels.capacity() * sizeof(int) + hnsw.offsets.capacity() * sizeof(size_t) +
                 hnsw.neighbors.capacity() * sizeof(faiss::HNSW::storage_idx_t);

    // vectors
    mem_usage += index_hnsw->storage->ntotal * index_hnsw->storage->d * sizeof(float);

    return mem_usage;
  }

  // IndexType::kFaissIvfPq
  if (index_type_ == IndexType::kFaissIvfPq) {
    auto* faiss_index = static_cast<faiss::Index*>(index_raw_);
    auto [transform, index_ivf_pq] = faiss_util::UnpackIvfPq(faiss_index);
    if (index_ivf_pq == nullptr) {
      T_LOG(WARNING)
          << "estimating memory usage for unsupported index types would always get result 1";
      return 1;
    }

    // IndexPreTransform
    if (transform != nullptr) {
      mem_usage += sizeof(*transform);
      // IndexPreTransform.VectorTransform
      mem_usage += transform->chain.capacity() * sizeof(faiss::VectorTransform*);
      for (auto chain_ptr : transform->chain) {
        mem_usage += sizeof(*chain_ptr);
      }
    }

    // IndexIvfPq
    mem_usage += sizeof(*index_ivf_pq);
    // IndexIvfPq.reconstruction_errors
    for (auto reconstruction_error : index_ivf_pq->reconstruction_errors) {
      mem_usage += sizeof(reconstruction_error);
      mem_usage += reconstruction_error.capacity() * sizeof(float);
    }

    // IndexIVFPQ.PolysemousTraining
    if (index_ivf_pq->polysemous_training != nullptr) {
      mem_usage += sizeof(*index_ivf_pq->polysemous_training);
    }

    // IndexIVF.InvertedLists
    if (index_ivf_pq->invlists != nullptr) {
      mem_usage += sizeof(*index_ivf_pq->invlists);
      mem_usage += index_ivf_pq->invlists->compute_ntotal() *
                   (index_ivf_pq->code_size + sizeof(faiss::Index::idx_t));
    }

    // IndexIVF.DirectMap
    {
      mem_usage += index_ivf_pq->direct_map.array.capacity() * sizeof(faiss::Index::idx_t);
      // 估算 unordered_map 占用内存大小
      auto& m = index_ivf_pq->direct_map.hashtable;
      mem_usage += (m.size() * (sizeof(faiss::Index::idx_t) + sizeof(faiss::Index::idx_t)) +
                    m.bucket_count() * (sizeof(void*) + sizeof(size_t))) *
                   1.5;
    }

    // TODO: Level1Quantizer.Index(quantizer)
    // TODO: Level1Quantizer.Index(clustering_index)
    return mem_usage;
  }

  T_LOG(ERROR) << "not implemented yet";
  return 1;
}

}  // namespace tenann