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

#include "tenann/builder/faiss_index_builder.h"

namespace tenann {
class FaissIndexBuilderWithBuffer : public FaissIndexBuilder {
 public:
  explicit FaissIndexBuilderWithBuffer(const IndexMeta& meta);
  virtual ~FaissIndexBuilderWithBuffer();

  T_FORBID_COPY_AND_ASSIGN(FaissIndexBuilderWithBuffer);
  T_FORBID_MOVE(FaissIndexBuilderWithBuffer);

  IndexBuilder& Flush() override;

  /// Configure an inter-batch cap on the in-memory row buffer
  /// (data_buffer_ + id_buffer_). After each Add() batch is merged, if the
  /// index is already trained and the buffered row count has reached |rows|,
  /// the buffer is drained into the underlying faiss index via a single
  /// batched add. This is a between-batch threshold, not a hard cap: a single
  /// Add() call that contributes more than |rows| rows will temporarily grow
  /// the buffer to existing_buffer + incoming_batch before the drain fires,
  /// and faiss's graph build step itself allocates memory proportional to the
  /// batch passed to add(). Indexes that still need training keep the full
  /// payload until Flush() (training requires the whole sample today; future
  /// work may extend this to also train on the first |rows| sample). Passing
  /// 0 disables the intermediate flush.
  IndexBuilder& SetFlushThresholdRows(size_t rows) override;

 protected:
  void Merge(const TypedSliceIterator<float>& input_row_iterator, const idx_t* row_ids);
  void AddRaw(const TypedSliceIterator<float>& input_row_iterator) override;
  void AddWithRowIds(const TypedSliceIterator<float>& input_row_iterator, const idx_t* row_ids) override;
  void AddWithRowIdsAndNullFlags(const TypedSliceIterator<float>& input_row_iterator, const idx_t* row_ids,
                                 const uint8_t* null_flags) override;

  // Drain data_buffer_ / id_buffer_ into the underlying index if the buffered
  // row count has grown past flush_threshold_rows_ and the index is already
  // trained. No-op for untrained indexes — training in Flush() needs the
  // whole sample.
  void MaybeFlushBuffer();

 protected:
  std::unique_ptr<TypedSliceIterator<float>> input_row_iterator_ = nullptr;
  const int64_t* row_id_ = nullptr;
  std::vector<float> data_buffer_;
  std::vector<int64_t> id_buffer_;
  bool is_vl_array_ = false;

  // 256K rows default (~128 MiB at dim=128, scales with dim). Tune via
  // SetFlushThresholdRows() — set 0 to disable intermediate flushing.
  static constexpr size_t kDefaultFlushThresholdRows = 256ULL * 1024;  // 262144
  size_t flush_threshold_rows_ = kDefaultFlushThresholdRows;
};

}  // namespace tenann