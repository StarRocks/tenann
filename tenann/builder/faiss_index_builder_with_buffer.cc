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

void FaissIndexBuilderWithBuffer::AddImpl(const std::vector<SeqView>& input_columns,
                                          const int64_t* row_ids, const uint8_t* null_flags) {
  auto input_seq_type = input_columns[0].seq_view_type;
  TypedArraySeqView<float> array_seq;
  TypedVlArraySeqView<float> vl_array_seq;

  bool is_vl_array = false;
  if (input_seq_type == SeqViewType::kArraySeqView) {
    array_seq = TypedArraySeqView<float>(input_columns[0].seq_view.array_seq_view);
  } else if (input_seq_type == SeqViewType::kVlArraySeqView) {
    vl_array_seq = TypedVlArraySeqView<float>(input_columns[0].seq_view.vl_array_seq_view);
    is_vl_array = true;
  }

  if (row_ids == nullptr && null_flags == nullptr) {
    is_vl_array ? AddRaw(vl_array_seq) : AddRaw(array_seq);
    return;
  }

  if (row_ids != nullptr && null_flags != nullptr) {
    is_vl_array ? AddWithRowIdsAndNullFlags(vl_array_seq, row_ids, null_flags)
                : AddWithRowIdsAndNullFlags(array_seq, row_ids, null_flags);
    return;
  }

  if (row_ids != nullptr && null_flags == nullptr) {
    is_vl_array ? AddWithRowIds(vl_array_seq, row_ids) : AddWithRowIds(array_seq, row_ids);
    return;
  }

  if (row_ids == nullptr && null_flags != nullptr) {
    T_LOG(ERROR) << "adding nullable data without rowids is not supported";
    return;
  }
}

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

    index_writer_->WriteIndex(index_ref_, index_save_path_, memory_only_);
  }
  CATCH_FAISS_ERROR;

  return *this;
}

void FaissIndexBuilderWithBuffer::Merge(const TypedArraySeqView<float>& input_column,
                                        const int64_t* row_ids) {
  if (inputs_live_longer_than_this_ == false) {
    data_buffer_.insert(data_buffer_.end(), input_column.data,
                        input_column.data + input_column.size * input_column.dim);
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
        data_buffer_.insert(data_buffer_.end(), input_column.data,
                            input_column.data + input_column.size * input_column.dim);
        if (row_ids != nullptr) {
          id_buffer_.assign(row_id_, row_id_ + array_seq_.size);
          id_buffer_.insert(id_buffer_.end(), row_ids, row_ids + input_column.size);
        }
      } else {
        data_buffer_.insert(data_buffer_.end(), input_column.data,
                            input_column.data + input_column.size * input_column.dim);
        if (row_ids != nullptr) {
          id_buffer_.insert(id_buffer_.end(), row_ids, row_ids + input_column.size);
        }
      }
    }
  }
}

void FaissIndexBuilderWithBuffer::Merge(const TypedVlArraySeqView<float>& input_column,
                                        const int64_t* row_ids) {
  if (inputs_live_longer_than_this_ == false) {
    data_buffer_.insert(data_buffer_.end(), input_column.data,
                        input_column.data + input_column.size * common_params_.dim);
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
        data_buffer_.assign(vl_array_seq_.data,
                            vl_array_seq_.data + vl_array_seq_.size * common_params_.dim);
        data_buffer_.insert(data_buffer_.end(), input_column.data,
                            input_column.data + input_column.size * common_params_.dim);
        if (row_ids != nullptr) {
          id_buffer_.assign(row_id_, row_id_ + vl_array_seq_.size);
          id_buffer_.insert(id_buffer_.end(), row_ids, row_ids + input_column.size);
        }
      } else {
        data_buffer_.insert(data_buffer_.end(), input_column.data,
                            input_column.data + input_column.size * common_params_.dim);
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
  is_vl_array_ = true;
  CheckDimension(input_column, common_params_.dim);
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
  is_vl_array_ = true;
  CheckDimension(input_column, common_params_.dim);
  auto faiss_index = GetFaissIndex();
  if (faiss_index->is_trained) {
    faiss_index->add_with_ids(input_column.size, input_column.data, row_ids);
  } else {
    Merge(input_column, row_ids);
  }
}

void FaissIndexBuilderWithBuffer::AddWithRowIdsAndNullFlags(
    const TypedArraySeqView<float>& input_column, const int64_t* row_ids,
    const uint8_t* null_flags) {
  auto faiss_index = GetFaissIndex();
  size_t i = 0;
  for (auto slice : input_column) {
    if (null_flags[i] == 0) {
      data_buffer_.insert(data_buffer_.end(), slice.data, slice.data + slice.size);
      id_buffer_.push_back(row_ids[i]);
    }
    i += 1;
  }
}

void FaissIndexBuilderWithBuffer::AddWithRowIdsAndNullFlags(
    const TypedVlArraySeqView<float>& input_column, const int64_t* row_ids,
    const uint8_t* null_flags) {
  is_vl_array_ = true;
  auto faiss_index = GetFaissIndex();
  size_t i = 0;
  for (auto slice : input_column) {
    if (null_flags[i] == 0) {
      T_LOG_IF(ERROR, slice.size != common_params_.dim)
          << "invalid size for vector " << i << " : expected " << common_params_.dim << " but got "
          << slice.size;
      data_buffer_.insert(data_buffer_.end(), slice.data, slice.data + slice.size);
      id_buffer_.push_back(row_ids[i]);
    }
    i += 1;
  }
}

}  // namespace tenann