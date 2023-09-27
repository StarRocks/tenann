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
#include "faiss/VectorTransform.h"
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

FaissIndexBuilder::FaissIndexBuilder(const IndexMeta& meta)
    : IndexBuilder(meta), transform_(nullptr) {
  CHECK_AND_GET_META(index_meta_, common, "dim", int, dim_);

  bool is_vector_normed = false;
  GET_META_OR_DEFAULT(index_meta_, common, "is_vector_normed", bool, is_vector_normed, false);

  int metric_type_value;
  CHECK_AND_GET_META(index_meta_, common, "metric_type", int, metric_type_value);
  metric_type_ = static_cast<MetricType>(metric_type_value);

  auto index_type = meta.index_type();

  if (index_type == IndexType::kFaissHnsw) {
    T_CHECK(metric_type_ == MetricType::kL2Distance ||
            metric_type_ == MetricType::kCosineSimilarity)
        << "only l2_distance and cosine_similarity are permitted as distance measures for faiss "
           "hnsw";

    if (metric_type_ == MetricType::kCosineSimilarity) {
      GET_META_OR_DEFAULT(index_meta_, common, "is_vector_normed", bool, is_vector_normed_, false);
      if (!is_vector_normed_) {
        transform_ = std::make_unique<faiss::NormalizationTransform>(dim_);
      }
    }
  } else {
    T_CHECK(metric_type_ == MetricType::kL2Distance)
        << "only l2_distance is supported for this type of index";
  }
}

FaissIndexBuilder::~FaissIndexBuilder(){};

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
                                     const int64_t* row_ids, const uint8_t* null_flags,
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
    T_LOG_IF(ERROR, !this->use_custom_row_id_ && (null_flags != nullptr))
        << "custom rowid is disabled, adding data with null flags is not supported";

    // check input parameters
    T_CHECK(input_columns.size() == 1);
    auto input_seq_type = input_columns[0].seq_view_type;
    T_CHECK(input_seq_type == SeqViewType::kArraySeqView ||
            input_seq_type == SeqViewType::kVlArraySeqView);
    T_CHECK(input_columns[0].seq_view.array_seq_view.elem_type == PrimitiveType::kFloatType ||
            input_columns[0].seq_view.vl_array_seq_view.elem_type == PrimitiveType::kFloatType);

    inputs_live_longer_than_this_ = inputs_live_longer_than_this;

    // add data to index
    AddImpl(input_columns, row_ids, null_flags);
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

  std::unique_ptr<TypedSliceIterator<float>> input_row_iterator = nullptr;
  if (input_seq_type == SeqViewType::kArraySeqView) {
    input_row_iterator =
        std::make_unique<TypedSliceIterator<float>>(input_columns[0].seq_view.array_seq_view);
  } else if (input_seq_type == SeqViewType::kVlArraySeqView) {
    CheckDimension(input_columns[0].seq_view.vl_array_seq_view, dim_);
    input_row_iterator =
        std::make_unique<TypedSliceIterator<float>>(input_columns[0].seq_view.vl_array_seq_view);
  }
  T_DCHECK_NOTNULL(input_row_iterator);

  if (row_ids == nullptr && null_flags == nullptr) {
    AddRaw(input_row_iterator->data(), input_row_iterator->size());
    return;
  }

  if (row_ids != nullptr && null_flags == nullptr) {
    AddWithRowIds(*input_row_iterator, row_ids);
  }

  if (row_ids != nullptr && null_flags != nullptr) {
    AddWithRowIdsAndNullFlags(*input_row_iterator, row_ids, null_flags);
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
  FaissIndexAddBatch(faiss_index, input_column.size, input_column.data, row_ids);
}

void FaissIndexBuilder::AddWithRowIds(const TypedVlArraySeqView<float>& input_column,
                                      const int64_t* row_ids) {
  CheckDimension(input_column, dim_);
  auto faiss_index = GetFaissIndex();
  FaissIndexAddBatch(faiss_index, input_column.size, input_column.data, row_ids);
}

void FaissIndexBuilder::AddWithRowIdsAndNullFlags(const TypedArraySeqView<float>& input_column,
                                                  const int64_t* row_ids,
                                                  const uint8_t* null_flags) {
  auto faiss_index = GetFaissIndex();
  idx_t i = 0;
  for (auto slice : input_column) {
    if (null_flags[i] == 0) {
      FaissIndexAddSingle(faiss_index, slice.data, row_ids + i);
    }
    i += 1;
  }
}

void FaissIndexBuilder::AddWithRowIdsAndNullFlags(const TypedVlArraySeqView<float>& input_column,
                                                  const int64_t* row_ids,
                                                  const uint8_t* null_flags) {
  auto faiss_index = GetFaissIndex();
  idx_t i = 0;
  for (auto slice : input_column) {
    if (null_flags[i] == 0) {
      T_LOG_IF(ERROR, slice.size != dim_)
          << "invalid size for vector " << i << " : expected " << dim_ << " but got " << slice.size;
      FaissIndexAddSingle(faiss_index, slice.data, row_ids + i);
    }
    i += 1;
  }
}

void FaissIndexBuilder::AddRaw(const float* base, int64_t num_rows) {}

void FaissIndexBuilder::AddWithRowIds(const TypedSliceIterator<float>& input_row_iterator,
                                      const int64_t* row_ids) {
  auto faiss_index = GetFaissIndex();
  FaissIndexAddBatch(faiss_index, input_row_iterator.size(), input_row_iterator.data(), row_ids);
}

void FaissIndexBuilder::AddWithRowIdsAndNullFlags(
    const TypedSliceIterator<float>& input_row_iterator, const int64_t* row_ids,
    const uint8_t* null_flags) {
  auto faiss_index = GetFaissIndex();
  input_row_iterator.ForEach([=](idx_t i, const float* slice_data, idx_t slice_length) {
    if (null_flags[i] == 0) {
      FaissIndexAddSingle(faiss_index, slice_data, row_ids + i);
    }
  });
}

void FaissIndexBuilder::FaissIndexAddBatch(faiss::Index* index, idx_t num_rows, const float* data,
                                           const int64_t* rowids) {
  const float* transformed_data = data;
  if (transform_ != nullptr) {
    auto tranformed_buffer = TransformBatch(num_rows, data);
    transformed_data = tranformed_buffer.data();
  }

  if (rowids != nullptr) {
    index->add_with_ids(num_rows, transformed_data, rowids);
  } else {
    index->add(num_rows, transformed_data);
  }
}

void FaissIndexBuilder::FaissIndexAddSingle(faiss::Index* index, const float* data,
                                            const int64_t* rowid) {
  const float* transformed_data = data;
  if (transform_ != nullptr) {
    auto tranformed_buffer = TransformSingle(data);
    transformed_data = tranformed_buffer.data();
  }

  if (rowid != nullptr) {
    index->add_with_ids(1, transformed_data, rowid);
  } else {
    index->add(1, transformed_data);
  }
}

std::vector<float> FaissIndexBuilder::TransformBatch(idx_t num_rows, const float* data) {
  std::vector<float> transform_buffer(num_rows * dim_);
  transform_->apply_noalloc(num_rows, data, transform_buffer.data());
  return transform_buffer;
}

std::vector<float> FaissIndexBuilder::TransformSingle(const float* data) {
  std::vector<float> transform_buffer(dim_);
  transform_->apply_noalloc(1, data, transform_buffer.data());
  return transform_buffer;
}

void FaissIndexBuilder::CheckDimension(const TypedVlArraySeqView<float>& input_column, int dim) {
  // check vector sizes
  idx_t i = 0;
  for (auto slice : input_column) {
    T_LOG_IF(ERROR, slice.size != dim)
        << "invalid size for vector " << i << " : expected " << dim << " but got " << slice.size;
    i += 1;
  }
}

void FaissIndexBuilder::CheckDimension(const TypedSliceIterator<float>& input_column, idx_t dim) {
  // check vector sizes
  input_column.ForEach([=](idx_t i, const float* slice_data, idx_t slice_lengh) {
    T_LOG_IF(ERROR, slice_lengh != dim)
        << "invalid size for vector " << i << " : expected " << dim << " but got " << slice_lengh;
  });
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
