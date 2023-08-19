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
#include "tenann/index/index_reader.h"
#include "tenann/index/index_writer.h"
#include "tenann/store/index_cache.h"
#include "tenann/store/index_meta.h"

namespace tenann {

class IndexBuilder {
 public:
  virtual ~IndexBuilder() = default;

  void Build(const std::vector<SeqView>& input_columns, int primary_key_column_index);

  /// Write index file.
  void WriteIndex(const std::string& path, bool write_index_cache = true);

  /** Setters */
  IndexBuilder& SetIndexWriter(IndexWriter* writer);

  IndexBuilder& SetIndexCache(IndexCache* cache);

  /** Getters */
  const IndexMeta& index_meta() const;

  IndexWriter* index_writer();

  const IndexWriter* index_writer() const;

  IndexRef index() const;

  IndexCache* index_cache();

  const IndexCache* index_cache() const;

  bool is_built();

 protected:
  virtual void BuildImpl(const std::vector<SeqView>& input_columns,
                         int primary_key_column_index) = 0;

  /* meta */
  IndexMeta index_meta_;

  /* index */
  IndexRef index_;
  bool is_built_;

  /* writer and cache */
  IndexWriter* index_writer_;
  IndexCache* index_cache_;

  /* statistics */
  // @TODO: add statistics
  // Stats stats_;
};

}  // namespace tenann