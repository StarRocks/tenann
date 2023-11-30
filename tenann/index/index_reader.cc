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

#include "tenann/index/index_reader.h"
#include "tenann/index/parameter_serde.h"

#include "index_reader.h"

namespace tenann {

IndexReader::IndexReader(const IndexMeta& meta) : index_meta_(meta) {
  FetchParameters(meta, &read_index_options_);
}

IndexReader::~IndexReader() = default;

const IndexMeta& IndexReader::index_meta() const { return index_meta_; }

IndexRef IndexReader::ReadIndex(const std::string& path) {
  if (read_index_options_.read_index_cache) {
    auto cache_key = !read_index_options_.custom_cache_key.empty() ? read_index_options_.custom_cache_key : path;
    T_LOG_IF(ERROR, index_cache_ == nullptr) << "index cache not set";
    if (read_index_options_.force_read_and_overwrite_cache) {
      return ForceReadIndexAndOverwriteCache(path, cache_key);
    } else {
      auto found = index_cache_->Lookup(cache_key, &cache_handle_);
      if (found) {
        return cache_handle_.index_ref();
      } else {
        return ForceReadIndexAndOverwriteCache(path, cache_key);
      }
    }
  } else {
    return ReadIndexFile(path);
  }
}

IndexRef IndexReader::ForceReadIndexAndOverwriteCache(const std::string& path, const std::string& cache_key) {
  IndexRef index_ref = ReadIndexFile(path);
  index_cache_->Insert(cache_key, index_ref, &cache_handle_);
  return index_ref;
}

IndexReader& IndexReader::SetIndexCache(IndexCache* cache) {
  T_CHECK_NOTNULL(cache);
  index_cache_ = cache;
  return *this;
}

IndexCache* IndexReader::index_cache() { return index_cache_; }

const IndexCache* IndexReader::index_cache() const { return index_cache_; }

}  // namespace tenann