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

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "tenann/common/macros.h"
#include "tenann/index/index.h"
#include "tenann/index/index_cache.h"
#include "tenann/store/lru_cache.h"

namespace tenann {

/**
 * @brief Default LRU-backed implementation of IndexCache.
 *
 * Index memory is owned by the cached IndexRef; eviction drops the last
 * reference and runs the deleter registered on Index construction.
 */
class DefaultIndexCache : public IndexCache {
 public:
  explicit DefaultIndexCache(size_t capacity);
  ~DefaultIndexCache() override;

  static DefaultIndexCache* GetGlobalInstance();

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
  [[nodiscard]] bool Lookup(const CacheKey& key, IndexCacheHandle* handle) override;

  /**
   * @brief Insert an index with key into this cache.
   *
   * Given handle will be set to valid reference.
   * This function is thread-safe, and when two clients insert two same key
   * concurrently, this function can assure that only one value is cached.
   *
   * The cache charge is derived from `ref->EstimateMemoryUsage()`.
   */
  void Insert(const CacheKey& key, IndexRef ref, IndexCacheHandle* handle) override;

  /**
   * @brief Atomic get-or-create with per-key single-flight loader.
   *
   * Fast-path: Lookup; on hit return true without running loader. On miss,
   * acquire a per-key lock, recheck, and call loader exactly once across
   * concurrent callers. On loader exception the entry is NOT cached and the
   * exception propagates; subsequent callers waiting on the per-key lock will
   * retry.
   */
  [[nodiscard]] bool GetOrCreate(const CacheKey& key, const IndexLoader& loader,
                                 IndexCacheHandle* handle) override;

  void SetCapacity(size_t capacity);

  bool AdjustCapacity(int64_t delta, size_t min_capacity = 0);

  json status() const;

  size_t memory_usage() const;

  size_t capacity();

  uint64_t lookup_count();

  uint64_t hit_count();

 private:
  std::shared_ptr<std::mutex> get_or_create_load_lock(const std::string& key);

  std::unique_ptr<Cache> cache_ = nullptr;

  // Per-key load locks used by GetOrCreate to enforce single-flight. Entries
  // are weak_ptrs; the live shared_ptr is held on the calling thread's stack
  // while the loader runs. Expired entries are reaped every kReapInterval
  // inserts to keep per-miss cost amortized O(1) under cold-start bursts.
  static constexpr uint32_t kReapInterval = 1024;
  std::mutex load_locks_mu_;
  std::unordered_map<std::string, std::weak_ptr<std::mutex>> load_locks_;
  uint32_t reap_counter_ = 0;
};

}  // namespace tenann
