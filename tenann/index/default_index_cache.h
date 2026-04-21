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
#include "tenann/index/index_cache.h"
#include "tenann/store/lru_cache.h"

namespace tenann {

/**
 * @brief Default LRU-backed implementation of IndexCache.
 *
 * Used by stress_tool, python_bindings, and the tenann unit tests. Production
 * integrations (StarRocks) inject their own IndexCache via SetGlobalIndexCache
 * with stronger single-flight and memory-tracker guarantees.
 *
 * Index memory is owned by the cached IndexRef; eviction drops the last
 * reference and runs the deleter registered on Index construction.
 */
class DefaultIndexCache : public IndexCache {
 public:
  explicit DefaultIndexCache(size_t capacity);
  ~DefaultIndexCache() override;

  static DefaultIndexCache* GetGlobalInstance();

  [[nodiscard]] bool Lookup(const CacheKey& key, IndexCacheHandle* handle) override;

  void Insert(const CacheKey& key, IndexRef ref, IndexCacheHandle* handle) override;

  // Lookup-then-loader-then-Insert. Does NOT provide single-flight across
  // concurrent callers — two threads missing on the same key each run the
  // loader, and the second Insert overwrites the first (wasted I/O but no
  // correctness issue). Callers that need strict single-flight must use their
  // own IndexCache implementation.
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
  std::unique_ptr<Cache> cache_ = nullptr;
};

}  // namespace tenann
