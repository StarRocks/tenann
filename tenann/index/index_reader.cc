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
#include "tenann/util/runtime_profile.h"
#include "tenann/util/stop_watch.h"

#include "index_reader.h"

namespace tenann {

IndexReader::IndexReader(const IndexMeta& meta) : index_meta_(meta) {
  FetchParameters(meta, &index_reader_options_);
}

IndexReader::~IndexReader() = default;

const IndexMeta& IndexReader::index_meta() const { return index_meta_; }

IndexRef IndexReader::ReadIndex(const std::string& path) {
  read_timing_stats_ = {};
  if (!index_reader_options_.cache_index_file) {
    return ReadIndexFile(path);
  }
  const auto& cache_key = !index_reader_options_.custom_cache_key.empty()
                              ? index_reader_options_.custom_cache_key
                              : path;
  T_CHECK(index_cache_ != nullptr)
      << "IndexCache not injected. "
      << "Call tenann::SetGlobalIndexCache() during process initialization "
      << "before constructing readers/searchers.";
  if (index_reader_options_.force_read_and_overwrite_cache) {
    return ForceReadIndexAndOverwriteCache(path, cache_key);
  }
  // GetOrCreate may or may not deduplicate concurrent cold misses depending
  // on the IndexCache implementation (SR's cache single-flights; the default
  // does not). Either way, the loader returns a valid IndexRef and Insert
  // makes it visible — duplicate loads waste I/O but stay correct.
  //
  // cache_lookup_ns covers the whole GetOrCreate window, so on a miss it
  // also includes the loader's read_file_ns / init_index_ns phases.
  {
    ScopedRawTimer<MonotonicStopWatch> timer(&read_timing_stats_.cache_lookup_ns);
    (void)index_cache_->GetOrCreate(
        cache_key,
        [this, &path]() -> IndexRef { return ReadIndexFile(path); },
        &cache_handle_);
  }
  return cache_handle_.index_ref();
}

IndexRef IndexReader::ForceReadIndexAndOverwriteCache(const std::string& path, const std::string& cache_key) {
  IndexRef index_ref = ReadIndexFile(path);
  if (index_ref == nullptr) {
    cache_handle_ = IndexCacheHandle();
    return nullptr;
  }
  index_cache_->Insert(cache_key, index_ref, &cache_handle_);
  return index_ref;
}

IndexReader& IndexReader::SetIndexCache(IndexCache* cache) {
  // nullptr is allowed: caching is opt-in via index_reader_options_. ReadIndex
  // T_CHECKs at use-time when cache_index_file is enabled.
  index_cache_ = cache;
  return *this;
}

IndexReader& IndexReader::SetFileReader(IndexFileReaderPtr reader) {
  file_reader_ = std::move(reader);
  return *this;
}

IndexCache* IndexReader::index_cache() { return index_cache_; }

const IndexCache* IndexReader::index_cache() const { return index_cache_; }

IndexFileReaderPtr IndexReader::file_reader() const { return file_reader_; }

}  // namespace tenann