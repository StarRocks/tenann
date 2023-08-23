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

#include "tenann/common/macros.h"
#include "tenann/index/index.h"
#include "tenann/store/lru_cache.h"

namespace tenann {

class IndexCacheEntry;

/**
 * @brief  Wrapper around Cache, and used for cache indexes.
 *
 * The actual memory of indexes are hold by the underlying index raw pointers.
 * This class caches these pointers and trigger the deletion action when a cache entry is evicted.
 */
class IndexCache {
 public:
  IndexCache(size_t capacity);
  ~IndexCache();

  static IndexCache* GetGlobalInstance();

  /**
   * @brief Lookup an index in the cache by CacheKey.
   *
   * If the index is found, the cache entry will be written into [[handle]].
   *
   * @param key cache key
   * @param handle handle to write
   * @return true if index found
   * @return false if index not found
   */
  bool Lookup(const CacheKey& key, IndexCacheEntry* handle);

  /**
   * @brief Insert an index with key into this cache.
   *
   *
   * Given handle will be set to valid reference.
   * This function is thread-safe, and when two clients insert two same key
   * concurrently, this function can assure that only one function is cached.
   *
   * @param key cache key
   * @param index index to cache
   * @param handle will be set to valid reference after writing the given index to cache
   */
  void Insert(const CacheKey& key, IndexRef index, IndexCacheEntry* handle);

  size_t memory_usage() const;

  void SetCapacity(size_t capacity);

  size_t capacity();

  uint64_t lookup_count();

  uint64_t hit_count();

  bool AdjustCapacity(int64_t delta, size_t min_capacity = 0);

 private:
  std::unique_ptr<Cache> cache_ = nullptr;
};

/**
 * @brief A handle for index cache entry.
 *
 * This class make it easy to handle cache entry.
 * Users don't need to release the obtained cache entry.
 * This class will release the cache entry when it is destroyed.
 */
class IndexCacheEntry {
 public:
  IndexCacheEntry();
  IndexCacheEntry(Cache* cache, Cache::Handle* handle);
  ~IndexCacheEntry();
  TNN_FORBID_COPY_AND_ASSIGN(IndexCacheEntry);

  IndexCacheEntry(IndexCacheEntry&& other) noexcept;
  IndexCacheEntry& operator=(IndexCacheEntry&& other) noexcept;

  Cache* cache() const;
  const Index* index() const;

 private:
  Cache* cache_ = nullptr;
  Cache::Handle* handle_ = nullptr;
};

}  // namespace tenann