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

#include "tenann/builder/index_builder.h"
#include "tenann/common/typed_seq_view.h"

namespace faiss {
class Index;
}

namespace tenann {

class FaissIndexBuilder : public IndexBuilder {
 public:
  explicit FaissIndexBuilder(const IndexMeta& meta);
  virtual ~FaissIndexBuilder() = default;

  T_FORBID_COPY_AND_ASSIGN(FaissIndexBuilder);
  T_FORBID_MOVE(FaissIndexBuilder);

  IndexBuilder& Open() override;

  IndexBuilder& Open(const std::string& index_save_path) override;

  IndexBuilder& Add(const std::vector<SeqView>& input_columns, const int64_t* row_ids = nullptr,
                    const uint8_t* null_flags = nullptr,
                    bool inputs_live_longer_than_this = false) override;

  IndexBuilder& Flush(bool write_index_cache = false,
                      const char* custom_cache_key = nullptr) override;

  void Close() override;

  bool is_opened() override;
  bool is_closed() override;

 protected:
  virtual IndexRef InitIndex() = 0;

  void PrepareProfile() override;

  void AddImpl(const std::vector<SeqView>& input_columns, const int64_t* row_ids = nullptr,
               const uint8_t* null_flags = nullptr);

  void AddRaw(const TypedArraySeqView<float>& input_column);
  void AddRaw(const TypedVlArraySeqView<float>& input_column);

  void AddWithRowIds(const TypedArraySeqView<float>& input_column, const int64_t* row_ids);
  void AddWithRowIds(const TypedVlArraySeqView<float>& input_column, const int64_t* row_ids);

  void AddWithNullFlags(const TypedArraySeqView<float>& input_column, const uint8_t* null_flags);
  void AddWithNullFlags(const TypedVlArraySeqView<float>& input_column, const uint8_t* null_flags);

  void AddWithRowIdsAndNullFlags(const TypedArraySeqView<float>& input_column,
                                 const int64_t* row_ids, const uint8_t* null_flags);
  void AddWithRowIdsAndNullFlags(const TypedVlArraySeqView<float>& input_column,
                                 const int64_t* row_ids, const uint8_t* null_flags);

  static void CheckDimension(const TypedVlArraySeqView<float>& input_column, int dim);

  faiss::Index* GetFaissIndex();

  void SetOpenState();
  void SetCloseState();
 protected:
  int dim_ = -1;
  MetricType metric_type_;

  bool memory_only_ = false;
  bool is_opened_ = false;
  bool is_closed_ = false;
  bool is_trained_ = false;

  RuntimeProfile::Counter* open_total_timer_ = nullptr;
  RuntimeProfile::Counter* add_total_timer_ = nullptr;
  RuntimeProfile::Counter* flush_total_timer_ = nullptr;
  RuntimeProfile::Counter* close_total_timer_ = nullptr;
};

}  // namespace tenann