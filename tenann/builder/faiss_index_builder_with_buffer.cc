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

IndexBuilder& FaissIndexBuilderWithBuffer::Flush() {
  try {
    T_SCOPED_TIMER(flush_total_timer_);

    T_LOG_IF(ERROR, !is_opened_) << "index builder has not been opened";
    T_LOG_IF(ERROR, index_ref_ == nullptr) << "index has not been built";

    if (GetFaissIndex()->is_trained == false) {
      bool with_row_ids = row_id_ != nullptr || id_buffer_.size() > 0;
      if (id_buffer_.size()) {
        row_id_ = id_buffer_.data();
      }

      if (data_buffer_.size()) {
        GetFaissIndex()->train(data_buffer_.size() / common_params_.dim, data_buffer_.data());
        if (with_row_ids) {
          GetFaissIndex()->add_with_ids(data_buffer_.size() / common_params_.dim,
                                        data_buffer_.data(), row_id_);
        } else {
          GetFaissIndex()->add(data_buffer_.size() / common_params_.dim, data_buffer_.data());
        }
      } else {
        GetFaissIndex()->train(input_row_iterator_->size(), input_row_iterator_->data());
        if (with_row_ids) {
          GetFaissIndex()->add_with_ids(input_row_iterator_->size(), input_row_iterator_->data(), row_id_);
        } else {
          GetFaissIndex()->add(input_row_iterator_->size(), input_row_iterator_->data());
        }
      }
    }

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
}

}  // namespace tenann