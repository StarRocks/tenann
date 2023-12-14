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

#include <memory>

#include "tenann/common/json.h"
#include "tenann/index/index.h"
#include "tenann/index/parameters.h"
#include "tenann/index/index_cache.h"

namespace tenann {

class IndexReader {
 public:
  explicit IndexReader(const IndexMeta& meta);
  virtual ~IndexReader();

  T_FORBID_DEFAULT_CTOR(IndexReader);
  T_FORBID_COPY_AND_ASSIGN(IndexReader);
  T_FORBID_MOVE(IndexReader);

  // Read index cache
  IndexRef ReadIndex(const std::string& path);

  // Read index file
  virtual IndexRef ReadIndexFile(const std::string& path) = 0;

  /** Setters */
  IndexReader& SetIndexCache(IndexCache* cache);

  /** Getters */
  const IndexMeta& index_meta() const;
  IndexCache* index_cache();
  const IndexCache* index_cache() const;

 protected:
  /// @brief index meta
  IndexMeta index_meta_;
  /// @brief read options
  ReadIndexOptions index_reader_options_;
  /// @brief cache
  IndexCache* index_cache_ = nullptr;
  /**
   * @brief Use this handle to maintain a reference to the cache entry.
   *        Otherwise the cache entry may be cleaned when the reference count decreases to 1.
   *
   */
  IndexCacheHandle cache_handle_;

  IndexRef ForceReadIndexAndOverwriteCache(const std::string& path, const std::string& cache_key);
};

using IndexReaderRef = std::shared_ptr<IndexReader>;

}  // namespace tenann
