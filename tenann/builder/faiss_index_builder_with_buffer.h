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

  IndexBuilder& Flush(bool write_index_cache = false,
                      const char* custom_cache_key = nullptr) override;

 protected:
  void Merge(const TypedArraySeqView<float>& input_column, const int64_t* row_ids = nullptr);
  void Merge(const TypedVlArraySeqView<float>& input_column, const int64_t* row_ids = nullptr);
  void AddRaw(const TypedArraySeqView<float>& input_column) override;
  void AddRaw(const TypedVlArraySeqView<float>& input_column) override;

  void AddWithRowIds(const TypedArraySeqView<float>& input_column, const int64_t* row_ids) override;
  void AddWithRowIds(const TypedVlArraySeqView<float>& input_column, const int64_t* row_ids) override;

  void AddWithRowIdsAndNullFlags(const TypedArraySeqView<float>& input_column,
                                 const int64_t* row_ids, const uint8_t* null_flags) override;
  void AddWithRowIdsAndNullFlags(const TypedVlArraySeqView<float>& input_column,
                                 const int64_t* row_ids, const uint8_t* null_flags) override;
 protected:
  TypedArraySeqView<float> array_seq_;
  TypedVlArraySeqView<float> vl_array_seq_;
  const int64_t* row_id_ = nullptr;
  std::vector<float> data_buffer_;
  std::vector<int64_t> id_buffer_;
  bool is_vl_array_ = false;
};

}  // namespace tenann