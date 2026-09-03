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
#include "faiss/IndexFlatCodes.h"
#include "faiss/IndexHNSW.h"
#include "faiss/IndexIDMap.h"
#include "faiss/IndexIVFPQ.h"
#include "faiss/IndexPQ.h"
#include "faiss/IndexScalarQuantizer.h"
#include "tenann/common/logging.h"
#include "tenann/index/index_ivfpq_reader.h"
#include "tenann/index/internal/faiss_index_util.h"

namespace tenann {

Index::~Index() noexcept {
  if (index_raw_ != nullptr) {
    deleter_(index_raw_);
  }
}

Index::Index(Index&& rhs) noexcept { std::swap(*this, rhs); }

Index::Index(void* index, IndexType index_type, const std::function<void(void*)>& deleter,
             size_t explicit_bytes) noexcept
    : index_raw_(index),
      index_type_(index_type),
      deleter_(deleter),
      explicit_bytes_(explicit_bytes) {}

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
  // Caller-supplied byte count (e.g. IVF-PQ per-list block buffer).
  // When set, it reflects the actual malloc'd size and is the source
  // of truth for cache-charge accounting.
  if (explicit_bytes_ > 0) {
    return explicit_bytes_;
  }

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
      mem_usage += index_id_map->id_map.capacity() * sizeof(faiss::idx_t);
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
                 hnsw.neighbors.size() * sizeof(faiss::HNSW::storage_idx_t);

    // vectors: charge the storage index' real code size. It is fp32 only for a flat
    // storage; the quantized HNSW variants keep compressed codes (SQ8: 1 byte per
    // dimension, SQ4: 0.5, PQ: m_pq bytes per vector), and charging fp32 for those
    // inflates the cache charge 4x to 32x.
    const auto* storage = index_hnsw->storage;
    size_t code_size = static_cast<size_t>(storage->d) * sizeof(float);
    if (const auto* flat_codes = dynamic_cast<const faiss::IndexFlatCodes*>(storage)) {
      code_size = flat_codes->code_size;
    }
    mem_usage += static_cast<size_t>(storage->ntotal) * code_size;

    // The PQ codebook is shared by all vectors, so it is not covered by the per-vector
    // code size above. It is fixed-size (2^nbits * dim floats), which makes it the
    // dominant term for a small segment at high dim.
    if (const auto* pq_storage = dynamic_cast<const faiss::IndexPQ*>(storage)) {
      mem_usage += pq_storage->pq.centroids.capacity() * sizeof(float);
      mem_usage += pq_storage->pq.sdc_table.capacity() * sizeof(float);
    }

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
    for (const auto& reconstruction_error : index_ivf_pq->reconstruction_errors) {
      mem_usage += sizeof(reconstruction_error);
      mem_usage += reconstruction_error.capacity() * sizeof(float);
    }

    // IndexIVFPQ.PolysemousTraining
    if (index_ivf_pq->polysemous_training != nullptr) {
      mem_usage += sizeof(*index_ivf_pq->polysemous_training);
    }

    // IndexIVF.InvertedLists
    if (index_ivf_pq->invlists != nullptr) {
      if (const auto* block_cache_invlists =
              dynamic_cast<const faiss::BlockCacheInvertedLists*>(index_ivf_pq->invlists)) {
        mem_usage += block_cache_invlists->EstimateMemoryUsage();
      } else {
        mem_usage += sizeof(*index_ivf_pq->invlists);
        mem_usage += index_ivf_pq->invlists->compute_ntotal() *
                     (index_ivf_pq->code_size + sizeof(faiss::idx_t));
      }
    }

    // IndexIVF.DirectMap
    {
      mem_usage += index_ivf_pq->direct_map.array.capacity() * sizeof(faiss::idx_t);
      // Estimate unordered_map memory usage
      auto& m = index_ivf_pq->direct_map.hashtable;
      mem_usage += (m.size() * (sizeof(faiss::idx_t) + sizeof(faiss::idx_t)) +
                    m.bucket_count() * (sizeof(void*) + sizeof(size_t))) *
                   1.5;
    }

    // TODO: Level1Quantizer.Index(quantizer)
    // TODO: Level1Quantizer.Index(clustering_index)
    return mem_usage;
  }

  // IndexType::kFaissIvfSq
  if (index_type_ == IndexType::kFaissIvfSq) {
    auto* faiss_index = static_cast<faiss::Index*>(index_raw_);
    auto [transform, index_ivf_sq] = faiss_util::UnpackIvfSq(faiss_index);
    if (index_ivf_sq == nullptr) {
      T_LOG(WARNING)
          << "estimating memory usage for unsupported index types would always get result 1";
      return 1;
    }

    if (transform != nullptr) {
      mem_usage += sizeof(*transform);
      mem_usage += transform->chain.capacity() * sizeof(faiss::VectorTransform*);
      for (auto chain_ptr : transform->chain) {
        mem_usage += sizeof(*chain_ptr);
      }
    }

    mem_usage += sizeof(*index_ivf_sq);
    if (index_ivf_sq->invlists != nullptr) {
      mem_usage += sizeof(*index_ivf_sq->invlists);
      mem_usage += index_ivf_sq->invlists->compute_ntotal() *
                   (index_ivf_sq->code_size + sizeof(faiss::idx_t));
    }
    mem_usage += index_ivf_sq->direct_map.array.capacity() * sizeof(faiss::idx_t);
    auto& direct_map = index_ivf_sq->direct_map.hashtable;
    mem_usage += (direct_map.size() * (sizeof(faiss::idx_t) + sizeof(faiss::idx_t)) +
                  direct_map.bucket_count() * (sizeof(void*) + sizeof(size_t))) *
                 1.5;
    return mem_usage;
  }

  T_LOG(WARNING) << "not implemented yet";
  return 1;
}

}  // namespace tenann
