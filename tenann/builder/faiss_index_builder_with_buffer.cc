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

FaissIndexBuilderWithBuffer::FaissIndexBuilderWithBuffer(const IndexMeta& meta) : FaissIndexBuilder(meta) {
}

IndexBuilder& FaissIndexBuilderWithBuffer::Flush(bool write_index_cache, const char* custom_cache_key) {
  try {
    T_SCOPED_TIMER(flush_total_timer_);

    T_LOG_IF(ERROR, !is_opened_) << "index builder has not been opened";
    T_LOG_IF(ERROR, is_closed_) << "index builder has already been closed";
    T_LOG_IF(ERROR, index_ref_ == nullptr) << "index has not been built";

    if (memory_only_) return *this;

    if (GetFaissIndex()->is_trained == false) {
      bool with_row_ids = row_id_ != nullptr || id_buffer_.size() > 0;
      if (id_buffer_.size()) {
        row_id_ = id_buffer_.data();
      }

      if (data_buffer_.size()) {
        GetFaissIndex()->train(data_buffer_.size() / dim_, data_buffer_.data());
        if (with_row_ids) {
          GetFaissIndex()->add_with_ids(data_buffer_.size() / dim_, data_buffer_.data(), row_id_);
        } else {
          GetFaissIndex()->add(data_buffer_.size() / dim_, data_buffer_.data());
        }
      } else if (is_vl_array_) {
        GetFaissIndex()->train(vl_array_seq_.size, vl_array_seq_.data);
        if (with_row_ids) {
          GetFaissIndex()->add_with_ids(vl_array_seq_.size, vl_array_seq_.data, row_id_);
        } else {
          GetFaissIndex()->add(vl_array_seq_.size, vl_array_seq_.data);
        }
      } else {
        GetFaissIndex()->train(array_seq_.size, array_seq_.data);
        if (with_row_ids) {
          GetFaissIndex()->add_with_ids(array_seq_.size, array_seq_.data, row_id_);
        } else {
          GetFaissIndex()->add(array_seq_.size, array_seq_.data);
        }
      }
    }

    // write index file
    index_writer_->WriteIndex(index_ref_, index_save_path_);
    // write index cache
    if (write_index_cache) {
      T_CHECK(index_cache_ != nullptr);
      std::string cache_key = custom_cache_key ? custom_cache_key : index_save_path_;
      IndexCacheHandle handle;
      index_cache_->Insert(cache_key, index_ref_, &handle);
    }
  }
  CATCH_FAISS_ERROR;

  return *this;
}

void FaissIndexBuilderWithBuffer::Merge(const TypedArraySeqView<float>& input_column, const int64_t* row_ids) {
  if (inputs_live_longer_than_this_ == false) {
    data_buffer_.insert(data_buffer_.end(), input_column.data, input_column.data + input_column.size * input_column.dim);
    if (row_ids != nullptr) {
      id_buffer_.insert(id_buffer_.end(), row_ids, row_ids + input_column.size);
    }
  } else {
    if (array_seq_.size == 0) {
      array_seq_ = input_column;
      if (row_ids != nullptr) {
        row_id_ = row_ids;
      }
    } else {
      if (data_buffer_.size() == 0) {
        T_CHECK(array_seq_.dim == input_column.dim);
        data_buffer_.assign(array_seq_.data, array_seq_.data + array_seq_.size * array_seq_.dim);
        data_buffer_.insert(data_buffer_.end(), input_column.data, input_column.data + input_column.size * input_column.dim);
        if (row_ids != nullptr) {
          id_buffer_.assign(row_id_, row_id_ + array_seq_.size);
          id_buffer_.insert(id_buffer_.end(), row_ids, row_ids + input_column.size);
        }
      } else {
        data_buffer_.insert(data_buffer_.end(), input_column.data, input_column.data + input_column.size * input_column.dim);
        if (row_ids != nullptr) {
          id_buffer_.insert(id_buffer_.end(), row_ids, row_ids + input_column.size);
        }
      }
    }
  }
}

void FaissIndexBuilderWithBuffer::Merge(const TypedVlArraySeqView<float>& input_column, const int64_t* row_ids) {
  if (inputs_live_longer_than_this_ == false) {
    data_buffer_.insert(data_buffer_.end(), input_column.data, input_column.data + input_column.size * dim_);
    if (row_ids != nullptr) {
      id_buffer_.insert(id_buffer_.end(), row_ids, row_ids + input_column.size);
    }
  } else {
    if (vl_array_seq_.size == 0) {
      vl_array_seq_ = input_column;
      if (row_ids != nullptr) {
        row_id_ = row_ids;
      }
    } else {
      if (data_buffer_.size() == 0) {
        data_buffer_.assign(vl_array_seq_.data, vl_array_seq_.data + vl_array_seq_.size * dim_);
        data_buffer_.insert(data_buffer_.end(), input_column.data, input_column.data + input_column.size * dim_);
        if (row_ids != nullptr) {
          id_buffer_.assign(row_id_, row_id_ + vl_array_seq_.size);
          id_buffer_.insert(id_buffer_.end(), row_ids, row_ids + input_column.size);
        }
      } else {
        data_buffer_.insert(data_buffer_.end(), input_column.data, input_column.data + input_column.size * dim_);
        if (row_ids != nullptr) {
          id_buffer_.insert(id_buffer_.end(), row_ids, row_ids + input_column.size);
        }
      }
    }
  }
}

void FaissIndexBuilderWithBuffer::AddRaw(const TypedArraySeqView<float>& input_column) {
  auto faiss_index = GetFaissIndex();
  if (faiss_index->is_trained) {
    faiss_index->add(input_column.size, input_column.data);
  } else {
    Merge(input_column);
  }
}

void FaissIndexBuilderWithBuffer::AddRaw(const TypedVlArraySeqView<float>& input_column) {
  CheckDimension(input_column, dim_);
  auto faiss_index = GetFaissIndex();
  if (faiss_index->is_trained) {
    faiss_index->add(input_column.size, input_column.data);
  } else {
    Merge(input_column);
  }
}

void FaissIndexBuilderWithBuffer::AddWithRowIds(const TypedArraySeqView<float>& input_column,
                                      const int64_t* row_ids) {
  auto faiss_index = GetFaissIndex();
  if (faiss_index->is_trained) {
    faiss_index->add_with_ids(input_column.size, input_column.data, row_ids);
  } else {
    Merge(input_column, row_ids);
  }
}

void FaissIndexBuilderWithBuffer::AddWithRowIds(const TypedVlArraySeqView<float>& input_column,
                                      const int64_t* row_ids) {
  CheckDimension(input_column, dim_);
  auto faiss_index = GetFaissIndex();
  if (faiss_index->is_trained) {
    faiss_index->add_with_ids(input_column.size, input_column.data, row_ids);
  } else {
    Merge(input_column, row_ids);
  }
}

void FaissIndexBuilderWithBuffer::AddWithNullFlags(const TypedArraySeqView<float>& input_column,
                                         const uint8_t* null_flags) {
  auto faiss_index = GetFaissIndex();
  size_t i = 0;
  for (auto slice : input_column) {
    if (null_flags[i] != 0) {
      data_buffer_.insert(data_buffer_.end(), slice.data, slice.data + slice.size);
    }
    i += 1;
  }
}

void FaissIndexBuilderWithBuffer::AddWithNullFlags(const TypedVlArraySeqView<float>& input_column,
                                         const uint8_t* null_flags) {
  auto faiss_index = GetFaissIndex();
  size_t i = 0;
  for (auto slice : input_column) {
    if (null_flags[i] != 0) {
      T_LOG_IF(ERROR, slice.size != dim_)
          << "invalid size for vector " << i << " : expected " << dim_ << " but got " << slice.size;
      data_buffer_.insert(data_buffer_.end(), slice.data, slice.data + slice.size);
    }
  }
  i += 1;
}

void FaissIndexBuilderWithBuffer::AddWithRowIdsAndNullFlags(const TypedArraySeqView<float>& input_column,
                                                  const int64_t* row_ids,
                                                  const uint8_t* null_flags) {
  auto faiss_index = GetFaissIndex();
  size_t i = 0;
  for (auto slice : input_column) {
    if (null_flags[i] != 0) {
      data_buffer_.insert(data_buffer_.end(), slice.data, slice.data + slice.size);
      id_buffer_.push_back(row_ids[i]);
    }
    i += 1;
  }
}

void FaissIndexBuilderWithBuffer::AddWithRowIdsAndNullFlags(const TypedVlArraySeqView<float>& input_column,
                                                  const int64_t* row_ids,
                                                  const uint8_t* null_flags) {
  auto faiss_index = GetFaissIndex();
  size_t i = 0;
  for (auto slice : input_column) {
    if (null_flags[i] != 0) {
      T_LOG_IF(ERROR, slice.size != dim_)
          << "invalid size for vector " << i << " : expected " << dim_ << " but got " << slice.size;
      data_buffer_.insert(data_buffer_.end(), slice.data, slice.data + slice.size);
      id_buffer_.push_back(row_ids[i]);
    }
    i += 1;
  }
}

}  // namespace tenann