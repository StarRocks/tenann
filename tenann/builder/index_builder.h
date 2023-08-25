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

namespace tenann {

class IndexBuilder {
 public:
  IndexBuilder() = default;
  virtual ~IndexBuilder() = default;

  T_FORBID_COPY_AND_ASSIGN(IndexBuilder);
  T_FORBID_MOVE(IndexBuilder);

  /// Build index and use the row number as primary key.
  IndexBuilder& Build(const std::vector<SeqView>& input_columns);

  /// Build index and use the given column marked by [[primary_key_column_index]] as primary key.
  IndexBuilder& BuildWithPrimaryKey(const std::vector<SeqView>& input_columns,
                                    int primary_key_column_index);

  /**
   * @brief Write the built index to a file. Optionally write the index to cache.
   *
   * @param path File path to write the index.
   * @param write_index_cache Whether to write the index to cache.
   * @param use_custom_cache_key Whether to use a custom cache key (default cache key is the given
   * path).
   * @param custom_cache_key Custom cache key to be used when use_custom_cache_key=true.
   * @return IndexBuilder& Reference to this IndexBuilder instance.
   */
  IndexBuilder& WriteIndex(const std::string& path, bool write_index_cache = false,
                           bool use_custom_cache_key = false,
                           const std::string& custom_cache_key = "");

  /** Setters */
  IndexBuilder& SetIndexWriter(IndexWriter* writer);

  IndexBuilder& SetIndexCache(IndexCache* cache);

  /** Getters */
  const IndexMeta& index_meta() const;

  IndexWriter* index_writer();

  const IndexWriter* index_writer() const;

  IndexRef index_ref() const;

  IndexCache* index_cache();

  const IndexCache* index_cache() const;

  bool is_built();

 protected:
  /// Use the given primary key column as vector id.
  virtual void BuildWithPrimaryKeyImpl(const std::vector<SeqView>& input_columns,
                                       int primary_key_column_index) = 0;

  /// Use the row number as vector id.
  virtual void BuildImpl(const std::vector<SeqView>& input_columns) = 0;

  /* meta */
  IndexMeta index_meta_;

  /* index */
  IndexRef index_ref_;
  bool is_built_;

  /* writer and cache */
  IndexWriter* index_writer_;
  IndexCache* index_cache_;

  /* statistics */
  // @TODO: add statistics
  // Stats stats_;
};

}  // namespace tenann