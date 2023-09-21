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

#include "faiss_index_builder.h"

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

FaissIndexBuilder::FaissIndexBuilder(const IndexMeta& meta) : IndexBuilder(meta) {
  CHECK_AND_GET_META(index_meta_, common, "dim", int, dim_);

  int metric_type_value;
  CHECK_AND_GET_META(index_meta_, common, "metric_type", int, metric_type_value);
  metric_type_ = static_cast<MetricType>(metric_type_value);
  T_CHECK(metric_type_ == MetricType::kL2Distance) << "only l2_distance is supported now";
}

IndexBuilder& FaissIndexBuilder::Open() {
  T_SCOPED_TIMER(open_total_timer_);
  T_LOG_IF(ERROR, is_opened_) << "index builder has already been opened";

  memory_only_ = true;
  index_save_path_ = "";
  index_ref_ = InitIndex();
  SetOpenState();
  return *this;
}

IndexBuilder& FaissIndexBuilder::Open(const std::string& path) {
  try {
    T_SCOPED_TIMER(open_total_timer_);
    T_LOG_IF(ERROR, is_opened_) << "index builder has already been opened";

    memory_only_ = false;
    index_save_path_ = path;
    index_ref_ = InitIndex();
    SetOpenState();
  }
  CATCH_FAISS_ERROR;

  return *this;
}

IndexBuilder& FaissIndexBuilder::Add(const std::vector<SeqView>& input_columns,
                                     const int64_t* row_ids, const uint8_t* null_map,
                                     bool inputs_live_longer_than_this) {
  try {
    T_SCOPED_TIMER(add_total_timer_);

    // check builder states
    T_LOG_IF(ERROR, !is_opened_) << "index builder has not been opened";
    T_LOG_IF(ERROR, is_closed_) << "index builder has already been closed";
    T_LOG_IF(ERROR, this->use_custom_row_id_ && (row_ids == nullptr))
        << "custom rowid is enabled, please add data with rowids";
    T_LOG_IF(ERROR, !this->use_custom_row_id_ && (row_ids != nullptr))
        << "custom rowid is disabled, adding data with rowids is not supported";

    // check input parameters
    T_CHECK(input_columns.size() == 1);
    auto input_seq_type = input_columns[0].seq_view_type;
    T_CHECK(input_seq_type == SeqViewType::kArraySeqView ||
            input_seq_type == SeqViewType::kVlArraySeqView);
    T_CHECK(input_columns[0].seq_view.array_seq_view.elem_type == PrimitiveType::kFloatType ||
            input_columns[0].seq_view.vl_array_seq_view.elem_type == PrimitiveType::kFloatType);

    // add data to index
    AddImpl(input_columns, row_ids, null_map);
  }
  CATCH_FAISS_ERROR;

  return *this;
}

IndexBuilder& FaissIndexBuilder::Flush(bool write_index_cache, const char* custom_cache_key) {
  try {
    T_SCOPED_TIMER(flush_total_timer_);

    T_LOG_IF(ERROR, !is_opened_) << "index builder has not been opened";
    T_LOG_IF(ERROR, is_closed_) << "index builder has already been closed";
    T_LOG_IF(ERROR, index_ref_ == nullptr) << "index has not been built";

    if (memory_only_) return *this;

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

void FaissIndexBuilder::Close() {
  // @TODO(petri): clear buffer
  T_LOG_IF(ERROR, !is_opened_) << "index builder has not been opened";
  SetCloseState();
}

bool FaissIndexBuilder::is_opened() { return is_opened_; }

bool FaissIndexBuilder::is_closed() { return is_closed_; }

void FaissIndexBuilder::PrepareProfile() {
  open_total_timer_ = T_ADD_TIMER(profile_, "OpenTotalTime");
  add_total_timer_ = T_ADD_TIMER(profile_, "AddTotalTime");
  flush_total_timer_ = T_ADD_TIMER(profile_, "FlushTotalTime");
  close_total_timer_ = T_ADD_TIMER(profile_, "CloseTotalTime");
}

void FaissIndexBuilder::AddImpl(const std::vector<SeqView>& input_columns, const int64_t* row_ids,
                                const uint8_t* null_flags) {
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

void FaissIndexBuilder::AddRaw(const TypedArraySeqView<float>& input_column) {
  auto faiss_index = GetFaissIndex();
  faiss_index->add(input_column.size, input_column.data);
}

void FaissIndexBuilder::AddRaw(const TypedVlArraySeqView<float>& input_column) {
  CheckDimension(input_column, dim_);
  auto faiss_index = GetFaissIndex();
  faiss_index->add(input_column.size, input_column.data);
}

void FaissIndexBuilder::AddWithRowIds(const TypedArraySeqView<float>& input_column,
                                      const int64_t* row_ids) {
  auto faiss_index = GetFaissIndex();
  faiss_index->add_with_ids(input_column.size, input_column.data, row_ids);
}

void FaissIndexBuilder::AddWithRowIds(const TypedVlArraySeqView<float>& input_column,
                                      const int64_t* row_ids) {
  CheckDimension(input_column, dim_);
  auto faiss_index = GetFaissIndex();
  faiss_index->add_with_ids(input_column.size, input_column.data, row_ids);
}

void FaissIndexBuilder::AddWithNullFlags(const TypedArraySeqView<float>& input_column,
                                         const uint8_t* null_flags) {
  auto faiss_index = GetFaissIndex();
  size_t i = 0;
  for (auto slice : input_column) {
    if (null_flags[i] == 0) {
      faiss_index->add(1, slice.data);
    }
    i += 1;
  }
}

void FaissIndexBuilder::AddWithNullFlags(const TypedVlArraySeqView<float>& input_column,
                                         const uint8_t* null_flags) {
  auto faiss_index = GetFaissIndex();
  size_t i = 0;
  for (auto slice : input_column) {
    if (null_flags[i] == 0) {
      T_LOG_IF(ERROR, slice.size != dim_)
          << "invalid size for vector " << i << " : expected " << dim_ << " but got " << slice.size;
      faiss_index->add(1, slice.data);
    }
  }
  i += 1;
}

void FaissIndexBuilder::AddWithRowIdsAndNullFlags(const TypedArraySeqView<float>& input_column,
                                                  const int64_t* row_ids,
                                                  const uint8_t* null_flags) {
  auto faiss_index = GetFaissIndex();
  size_t i = 0;
  for (auto slice : input_column) {
    if (null_flags[i] == 0) {
      faiss_index->add_with_ids(1, slice.data, row_ids + i);
    }
    i += 1;
  }
}

void FaissIndexBuilder::AddWithRowIdsAndNullFlags(const TypedVlArraySeqView<float>& input_column,
                                                  const int64_t* row_ids,
                                                  const uint8_t* null_flags) {
  auto faiss_index = GetFaissIndex();
  size_t i = 0;
  for (auto slice : input_column) {
    if (null_flags[i] == 0) {
      T_LOG_IF(ERROR, slice.size != dim_)
          << "invalid size for vector " << i << " : expected " << dim_ << " but got " << slice.size;
      faiss_index->add_with_ids(1, slice.data, row_ids + i);
    }
    i += 1;
  }
}

void FaissIndexBuilder::CheckDimension(const TypedVlArraySeqView<float>& input_column, int dim) {
  // check vector sizes
  size_t i = 0;
  for (auto slice : input_column) {
    T_LOG_IF(ERROR, slice.size != dim)
        << "invalid size for vector " << i << " : expected " << dim << " but got " << slice.size;
    i += 1;
  }
}

faiss::Index* FaissIndexBuilder::GetFaissIndex() {
  return static_cast<faiss::Index*>(index_ref_->index_raw());
}

void FaissIndexBuilder::SetOpenState() {
  is_opened_ = true;
  is_closed_ = false;
}

void FaissIndexBuilder::SetCloseState() {
  is_opened_ = false;
  is_closed_ = false;
}

}  // namespace tenann
