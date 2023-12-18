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

 protected:
  void Merge(const TypedSliceIterator<float>& input_row_iterator, const idx_t* row_ids);
  void AddRaw(const TypedSliceIterator<float>& input_row_iterator) override;
  void AddWithRowIds(const TypedSliceIterator<float>& input_row_iterator, const idx_t* row_ids) override;
  void AddWithRowIdsAndNullFlags(const TypedSliceIterator<float>& input_row_iterator, const idx_t* row_ids,
                                 const uint8_t* null_flags) override;

 protected:
  std::unique_ptr<TypedSliceIterator<float>> input_row_iterator_ = nullptr;
  const int64_t* row_id_ = nullptr;
  std::vector<float> data_buffer_;
  std::vector<int64_t> id_buffer_;
  bool is_vl_array_ = false;
};

}  // namespace tenann