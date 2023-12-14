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

#include "tenann/index/index_writer.h"
#include "tenann/index/parameter_serde.h"
#include "tenann/common/logging.h"

#include "index_writer.h"

namespace tenann {

IndexWriter::IndexWriter(const IndexMeta& meta) : index_meta_(meta) {
  FetchParameters(meta, &index_writer_options_);
}

IndexWriter::~IndexWriter() = default;

const IndexMeta& IndexWriter::index_meta() const { return index_meta_; }

void IndexWriter::WriteIndex(IndexRef index, const std::string& path, bool memory_only) {
  if (index_writer_options_.write_index_cache) {
    T_LOG_IF(ERROR, index_cache_ == nullptr) << "index cache not set";
    const std::string& cache_key = !index_writer_options_.custom_cache_key.empty() ? index_writer_options_.custom_cache_key : path;
    IndexCacheHandle handle;
    index_cache_->Insert(cache_key, index, &handle);
  }
  if (memory_only) {
    return;
  }

  WriteIndexFile(index, path);
}

IndexWriter& IndexWriter::SetIndexCache(IndexCache* cache) {
  T_CHECK_NOTNULL(cache);
  index_cache_ = cache;
  return *this;
}

IndexCache* IndexWriter::index_cache() { return index_cache_; }

const IndexCache* IndexWriter::index_cache() const { return index_cache_; }

}  // namespace tenann