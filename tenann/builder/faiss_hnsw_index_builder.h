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
#include <vector>

#include "tenann/builder/faiss_index_builder_with_buffer.h"
#include "tenann/index/parameters.h"

namespace faiss {
class IndexHNSW;
}

namespace tenann {

class HnswBuildStorage;

/// Index builder for the faiss HNSW family, including HNSWFlat (no
/// quantization) and quantized variants (HNSWSQ4/SQ8/PQ).
///
/// Rows reach faiss in batches, so faiss's hnsw_add_vertices can engage its
/// OpenMP parallel region (guarded by n > 100) instead of inserting per row.
///
/// HNSWFlat stages Add() rows straight into the index's own storage, with no
/// separate buffer or transform copy, and builds the graph for every
/// SetFlushThresholdRows() rows and at Flush(). Rows join the graph, and
/// ntotal, only then. Building the graph is part of FlushTotalTime.
///
/// Quantized variants require training before adding vectors; the parent
/// (FaissIndexBuilderWithBuffer) buffers Add() input until Flush(), then
/// trains on the buffered data before adding. Peak buffer memory is bounded by
/// SetFlushThresholdRows().
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

  IndexBuilder& Flush() override;

 protected:
  IndexRef InitIndex() override;

  void AddRaw(const TypedSliceIterator<float>& input_row_iterator) override;
  void AddWithRowIds(const TypedSliceIterator<float>& input_row_iterator,
                     const idx_t* row_ids) override;
  void AddWithRowIdsAndNullFlags(const TypedSliceIterator<float>& input_row_iterator,
                                 const idx_t* row_ids, const uint8_t* null_flags) override;

 private:
  // Flat storage: rows are written straight into the index's storage (normalized there for cosine),
  // and the graph is built for every flush_threshold_rows_ of them, so no row buffer or transform
  // copy exists.
  void StageRow(const float* row, const idx_t* row_id);
  void BuildStaged();
  void CheckGraphIntact() const;

  FaissHnswIndexParams index_params_;
  FaissHnswSearchParams search_params_;

  HnswBuildStorage* build_storage_ = nullptr;  // owned by the index; null for quantized storage
  bool normalize_rows_ = false;
  std::vector<idx_t> staged_ids_;
  // Set while a graph build mutates the index; still set afterwards only if that build threw,
  // leaving the graph, storage and IDMap out of step, so the index must not be added to or written.
  bool graph_broken_ = false;
};

}  // namespace tenann
