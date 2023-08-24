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

#include "tenann/builder/index_builder.h"

#include "tenann/common/logging.h"

namespace tenann {

IndexBuilder& IndexBuilder::Build(const std::vector<SeqView>& input_columns) {
  // @TODO: collect some statistics for the building procedure
  BuildImpl(input_columns);
  return *this;
}

IndexBuilder& IndexBuilder::BuildWithPrimaryKey(const std::vector<SeqView>& input_columns,
                                                int primary_key_column_index) {
  // @TODO: collect some statistics for the building procedure
  BuildWithPrimaryKeyImpl(input_columns, primary_key_column_index);
  return *this;
}

IndexBuilder& IndexBuilder::WriteIndex(const std::string& path, bool write_index_cache,
                                       bool use_custom_cache_key,
                                       const std::string& custom_cache_key) {
  T_DCHECK_NOTNULL(index_ref_);
  // write index file
  index_writer_->WriteIndex(index_ref_, path);
  // write index cache
  if (write_index_cache) {
    auto cache_key = use_custom_cache_key ? custom_cache_key : path;
    T_DCHECK_NOTNULL(index_cache_);
    IndexCacheHandle handle;
    index_cache_->Insert(cache_key, index_ref_, &handle);
  }
  return *this;
}

IndexBuilder& IndexBuilder::SetIndexWriter(IndexWriter* writer) {
  index_writer_ = writer;
  return *this;
}

IndexBuilder& IndexBuilder::SetIndexCache(IndexCache* cache) {
  T_DCHECK_NOTNULL(cache);
  index_cache_ = cache;
  return *this;
}

const IndexMeta& IndexBuilder::index_meta() const { return index_meta_; }

IndexRef IndexBuilder::index_ref() const { return index_ref_; }

IndexWriter* IndexBuilder::index_writer() { return index_writer_; }

const IndexWriter* IndexBuilder::index_writer() const { return index_writer_; }

IndexCache* IndexBuilder::index_cache() { return index_cache_; }

const IndexCache* IndexBuilder::index_cache() const { return index_cache_; }

bool IndexBuilder::is_built() { return is_built_; }

}  // namespace tenann