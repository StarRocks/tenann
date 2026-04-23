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

#include <cstddef>

#include "tenann/builder/faiss_index_builder_with_buffer.h"
#include "tenann/index/parameters.h"

namespace faiss {
class IndexHNSW;
}

namespace tenann {

/// Index builder for the faiss HNSW family, including HNSWFlat (no
/// quantization) and quantized variants (HNSWSQ4/SQ8/PQ).
///
/// Inherits from FaissIndexBuilderWithBuffer so that rows added via Add() are
/// buffered and then dispatched to faiss in a single add(N, ...) call from
/// Flush(). Batch-add lets faiss's hnsw_add_vertices engage its OpenMP
/// parallel region (guarded by n > 100), giving multi-core speedup over
/// per-row insertion. Peak buffer memory is bounded by SetFlushThresholdRows().
///
/// Quantized variants require training before adding vectors; the parent
/// buffers Add() input until Flush(), then trains on the buffered data before
/// adding. For HNSWFlat the underlying faiss index reports `is_trained == true`
/// and the train phase in Flush() is a no-op.
class FaissHnswIndexBuilder final : public FaissIndexBuilderWithBuffer {
 public:
  explicit FaissHnswIndexBuilder(const IndexMeta& meta);
  virtual ~FaissHnswIndexBuilder();

  T_FORBID_COPY_AND_ASSIGN(FaissHnswIndexBuilder);
  T_FORBID_MOVE(FaissHnswIndexBuilder);

  /// Minimum number of training rows required by the configured quantizer.
  ///   Flat:        0  (no training)
  ///   SQ4 / SQ8:   1  (per-dimension scalar quantizer; faiss accepts very
  ///                    small training sets but requires at least one row)
  ///   PQ:          (1 << nbits_pq) * 100  (faiss recommendation)
  ///
  /// Callers (e.g. async build in StarRocks) can use this to decide whether
  /// to skip building and fall back to brute-force when too few rows are
  /// available.
  size_t GetMinTrainRows() const;

 protected:
  IndexRef InitIndex() override;

 private:
  FaissHnswIndexParams index_params_;
  FaissHnswSearchParams search_params_;
};

}  // namespace tenann
