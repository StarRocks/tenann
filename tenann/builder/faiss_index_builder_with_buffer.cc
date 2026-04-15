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

#include "tenann/builder/faiss_index_builder_with_buffer.h"

#include <sstream>

#include "faiss/IndexHNSW.h"
#include "faiss/IndexIDMap.h"
#include "faiss/index_factory.h"
#include "faiss_hnsw_index_builder.h"
#include "fmt/format.h"
#include "tenann/builder/faiss_hnsw_index_builder.h"
#include "tenann/common/logging.h"
#include "tenann/common/typed_seq_view.h"
#include "tenann/index/index.h"
#include "tenann/util/runtime_profile.h"
#include "tenann/util/runtime_profile_macros.h"

namespace tenann {

FaissIndexBuilderWithBuffer::FaissIndexBuilderWithBuffer(const IndexMeta& meta)
    : FaissIndexBuilder(meta) {}

FaissIndexBuilderWithBuffer::~FaissIndexBuilderWithBuffer(){};

IndexBuilder& FaissIndexBuilderWithBuffer::SetFlushThresholdRows(size_t rows) {
  flush_threshold_rows_ = rows;
  return *this;
}

// Drain data_buffer_ / id_buffer_ into the underlying faiss index if:
//   (a) the index is already trained (Flat/HNSWFlat — SQ/PQ/IVFPQ still need
//       the whole sample for training, see Flush),
//   (b) intermediate flushing is not disabled (flush_threshold_rows_ > 0),
//   (c) the buffered row count has reached the configured cap.
// Then clear the buffers so the next batch starts fresh. This bounds peak
// memory during long ingest streams while keeping batches large enough to
// engage faiss's OpenMP region inside hnsw_add_vertices (guarded by n > 100).
void FaissIndexBuilderWithBuffer::MaybeFlushBuffer() {
  if (flush_threshold_rows_ == 0) return;
  if (!GetFaissIndex()->is_trained) return;
  idx_t n = data_buffer_.size() / common_params_.dim;
  if (n < static_cast<idx_t>(flush_threshold_rows_)) return;

  if (!id_buffer_.empty()) {
    GetFaissIndex()->add_with_ids(n, data_buffer_.data(), id_buffer_.data());
  } else if (row_id_ != nullptr) {
    GetFaissIndex()->add_with_ids(n, data_buffer_.data(), row_id_);
  } else {
    GetFaissIndex()->add(n, data_buffer_.data());
  }
  data_buffer_.clear();
  id_buffer_.clear();
}

// Flush contract for future maintainers:
//
//  1. The buffer drain (phase 2 below) is unconditional — do NOT move it back
//     under an is_trained guard. A pre-trained index (e.g. HNSWFlat) routed
//     through this class would otherwise silently drop its buffered rows.
//  2. Intermediate flushing (MaybeFlushBuffer) only fires once the index is
//     trained, and only for the copy path (data_buffer_). The zero-copy path
//     (input_row_iterator_, enabled via inputs_live_longer_than_this_ = true)
//     is never drained early — the caller promised us the input outlives the
//     builder, so we wait for Flush.
//  3. Phase 2 fires a single add(N, ...). A caller that buffers a very large
//     batch without calling SetFlushThresholdRows will incur a matching
//     memory spike inside faiss during graph build.
IndexBuilder& FaissIndexBuilderWithBuffer::Flush() {
  try {
    T_SCOPED_TIMER(flush_total_timer_);

    T_LOG_IF(ERROR, !is_opened_) << "index builder has not been opened";
    T_LOG_IF(ERROR, index_ref_ == nullptr) << "index has not been built";

    bool with_row_ids = row_id_ != nullptr || id_buffer_.size() > 0;
    if (id_buffer_.size()) {
      row_id_ = id_buffer_.data();
    }

    // Phase 1: train (only when the index requires it).
    //   - Flat / HNSWFlat: is_trained == true at construction -> skip.
    //   - SQ / PQ / IVFPQ: is_trained == false -> train on the buffered sample.
    if (GetFaissIndex()->is_trained == false) {
      if (data_buffer_.size()) {
        GetFaissIndex()->train(data_buffer_.size() / common_params_.dim, data_buffer_.data());
      } else if (input_row_iterator_) {
        GetFaissIndex()->train(input_row_iterator_->size(), input_row_iterator_->data());
      }
    }

    // Phase 2: batch add (always runs, independent of is_trained).
    //   - Single add(N, ...) / add_with_ids(N, ...) per Flush: required for
    //     faiss hnsw_add_vertices to enter its OpenMP parallel region.
    //   - Unconditional: guarantees rows reach the index for both
    //     pre-trained (HNSWFlat) and trained-on-flush (SQ/PQ/IVFPQ) builders.
    if (data_buffer_.size()) {
      idx_t n = data_buffer_.size() / common_params_.dim;
      if (with_row_ids) {
        GetFaissIndex()->add_with_ids(n, data_buffer_.data(), row_id_);
      } else {
        GetFaissIndex()->add(n, data_buffer_.data());
      }
    } else if (input_row_iterator_) {
      if (with_row_ids) {
        GetFaissIndex()->add_with_ids(input_row_iterator_->size(), input_row_iterator_->data(), row_id_);
      } else {
        GetFaissIndex()->add(input_row_iterator_->size(), input_row_iterator_->data());
      }
    }

    // Release the buffered rows once they have been added. This also makes
    // Flush idempotent: a second call is a no-op rather than a double add.
    data_buffer_.clear();
    data_buffer_.shrink_to_fit();
    id_buffer_.clear();
    id_buffer_.shrink_to_fit();

    index_writer_->WriteIndex(index_ref_, index_save_path_, memory_only_);
  }
  CATCH_FAISS_ERROR;

  return *this;
}

void FaissIndexBuilderWithBuffer::Merge(const TypedSliceIterator<float>& input_row_iterator,
                                        const idx_t* row_ids) {
  if (inputs_live_longer_than_this_ == false) {
    data_buffer_.insert(data_buffer_.end(), input_row_iterator.data(),
                        input_row_iterator.data() + input_row_iterator.size() * common_params_.dim);
    if (row_ids != nullptr) {
      id_buffer_.insert(id_buffer_.end(), row_ids, row_ids + input_row_iterator.size());
    }
  } else {
    if (input_row_iterator_ == nullptr) {
      input_row_iterator_ = std::make_unique<TypedSliceIterator<float>>(input_row_iterator);
      if (row_ids != nullptr) {
        row_id_ = row_ids;
      }
    } else {
      if (data_buffer_.size() == 0) {
        data_buffer_.assign(input_row_iterator_->data(), input_row_iterator_->data() + input_row_iterator_->size() * common_params_.dim);
        data_buffer_.insert(data_buffer_.end(), input_row_iterator.data(),
                            input_row_iterator.data() + input_row_iterator.size() * common_params_.dim);
        if (row_ids != nullptr) {
          id_buffer_.assign(row_id_, row_id_ + input_row_iterator_->size());
          id_buffer_.insert(id_buffer_.end(), row_ids, row_ids + input_row_iterator.size());
        }
      } else {
        data_buffer_.insert(data_buffer_.end(), input_row_iterator.data(),
                            input_row_iterator.data() +input_row_iterator.size() * common_params_.dim);
        if (row_ids != nullptr) {
          id_buffer_.insert(id_buffer_.end(), row_ids, row_ids + input_row_iterator.size());
        }
      }
    }
  }
  MaybeFlushBuffer();
}

void FaissIndexBuilderWithBuffer::AddRaw(const TypedSliceIterator<float>& input_row_iterator) {
  auto faiss_index = GetFaissIndex();
  if (faiss_index->is_trained) {
    faiss_index->add(input_row_iterator.size(), input_row_iterator.data());
  } else {
    Merge(input_row_iterator, nullptr);
  }
}

void FaissIndexBuilderWithBuffer::AddWithRowIds(const TypedSliceIterator<float>& input_row_iterator,
                                      const idx_t* row_ids) {
  auto faiss_index = GetFaissIndex();
  if (faiss_index->is_trained) {
    FaissIndexAddBatch(faiss_index, input_row_iterator.size(), input_row_iterator.data(), row_ids);
  } else {
    Merge(input_row_iterator, row_ids);
  }
}

void FaissIndexBuilderWithBuffer::AddWithRowIdsAndNullFlags(
    const TypedSliceIterator<float>& input_row_iterator, const idx_t* row_ids,
    const uint8_t* null_flags) {
  auto faiss_index = GetFaissIndex();
  input_row_iterator.ForEach([=](idx_t i, const float* slice_data, idx_t slice_length) {
    if (null_flags[i] == 0) {
      data_buffer_.insert(data_buffer_.end(), slice_data, slice_data + slice_length);
      id_buffer_.push_back(row_ids[i]);
    }
  });
  MaybeFlushBuffer();
}

}  // namespace tenann