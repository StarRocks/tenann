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

#include "tenann/common/seq_view.h"
#include "tenann/index/index_cache.h"
#include "tenann/index/index_reader.h"
#include "tenann/index/index_writer.h"
#include "tenann/store/index_meta.h"
#include "tenann/util/runtime_profile.h"

namespace tenann {

/**
 * @brief Super class for all index builders. Not thread-safe.
 *
 */
class IndexBuilder {
 public:
  explicit IndexBuilder(const IndexMeta& meta);
  virtual ~IndexBuilder();

  T_FORBID_DEFAULT_CTOR(IndexBuilder);
  T_FORBID_COPY_AND_ASSIGN(IndexBuilder);
  T_FORBID_MOVE(IndexBuilder);

  /// Open an in-memory index builder.
  virtual IndexBuilder& Open() = 0;

  /// Open a disk-based index builder with the specified path for index file or directory.
  virtual IndexBuilder& Open(const std::string& index_save_path) = 0;

  /**
   * @brief Inserts a batch of data into the index.
   *
   * @param input_columns The columns to be indexed.
   * @param row_ids       Optional custom row IDs (by default, the row number is used as the row ID,
   *                      where the first row has row ID 0, the second row has row ID 1, and so on).
   * @param null_flags    Optional null map.
   * @param inputs_live_longer_than_this
   *                      Indicates whether the lifetimes of the input pointers are long enough
   * (compared to the IndexBuilder). If not, for certain indexes, the function may need to
   * make a copy of the data for subsequent index construction. If the lifetimes are long enough,
   * maintaining references is sufficient, and data copying is not necessary.
   *
   * @return IndexBuilder&
   */
  virtual IndexBuilder& Add(const std::vector<SeqView>& input_columns,
                            const int64_t* row_ids = nullptr, const uint8_t* null_flags = nullptr,
                            bool inputs_live_longer_than_this = false) = 0;

  /**
   * @brief Completes the construction of the index and forces the index structure to be
   * flushed to disk or memory.
   *
   * The first flush performs the initial training and construction of the index.
   * Once the first flush is completed, the index builder can also support adding new data until it
   * is closed.
   *
   * @param write_index_cache
   * @param custom_cache_key
   *
   * @return IndexBuilder&
   */
  virtual IndexBuilder& Flush(bool write_index_cache = false,
                              const char* custom_cache_key = nullptr) = 0;

  // Clean resources and close this builder.
  virtual void Close() = 0;

  virtual bool is_opened() = 0;
  virtual bool is_closed() = 0;

  /** Setters */
  IndexBuilder& SetBuildOptions(const json& options);
  IndexBuilder& SetIndexWriter(IndexWriterRef writer);
  IndexBuilder& SetIndexCache(IndexCache* cache);
  IndexBuilder& EnableCustomRowId();
  IndexBuilder& EnableProfile();
  IndexBuilder& DisableProfile();

  /** Getters */
  const IndexMeta& index_meta() const;

  IndexWriter* index_writer();

  const IndexWriter* index_writer() const;

  IndexRef index_ref() const;

  IndexCache* index_cache();

  const IndexCache* index_cache() const;

  RuntimeProfile* profile();

 protected:
  virtual void PrepareProfile() = 0;

  /* meta */
  IndexMeta index_meta_;
  /* index */
  IndexRef index_ref_;
  /* options */
  json build_options_;
  bool use_custom_row_id_ = false;

  /* writer and cache */
  IndexWriterRef index_writer_ = nullptr;
  IndexCache* index_cache_ = nullptr;
  std::string index_save_path_ = "";

  /* statistics */
  std::unique_ptr<RuntimeProfile> profile_ = nullptr;
};

}  // namespace tenann