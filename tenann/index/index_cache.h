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

#include "tenann/common/macros.h"
#include "tenann/index/index.h"
#include "tenann/store/lru_cache.h"  // CacheKey

namespace tenann {

class IndexCacheHandle;

class IndexCache {
 public:
  using IndexLoader = std::function<IndexRef()>;

  virtual ~IndexCache() = default;

  // Lookup a key. On hit, fills handle with a pinned reference. Returns true on hit.
  [[nodiscard]] virtual bool Lookup(const CacheKey& key, IndexCacheHandle* handle) = 0;

  // Insert ref. Implementation calls ref->EstimateMemoryUsage() to determine charge.
  virtual void Insert(const CacheKey& key, IndexRef ref, IndexCacheHandle* handle) = 0;

  // Atomic get-or-create. Concurrent callers for the same key run loader at most
  // once. Returns true if fast-path hit (cache had the entry or a concurrent
  // caller finished loading before this caller acquired the per-key lock);
  // returns false if this caller ran loader itself.
  //
  // On loader exception, the exception propagates to the caller currently
  // executing loader. The entry is NOT cached. Other callers blocked on the
  // per-key lock will retry and re-run loader. `handle` is unmodified on
  // exception; return value is undefined.
  [[nodiscard]] virtual bool GetOrCreate(const CacheKey& key, const IndexLoader& loader,
                                         IndexCacheHandle* handle) = 0;
};

// Lifetime: IndexCacheHandle must be destroyed before the IndexCache
// it was obtained from. The releaser_ destructor may call back into the cache
// (e.g. SR's VectorIndexCache calls Cache::release), which would be a
// use-after-free if the cache is gone.
class IndexCacheHandle {
 public:
  IndexCacheHandle() = default;
  IndexCacheHandle(IndexRef ref, std::shared_ptr<void> releaser)
      : ref_(std::move(ref)), releaser_(std::move(releaser)) {}

  T_FORBID_COPY_AND_ASSIGN(IndexCacheHandle);

  IndexCacheHandle(IndexCacheHandle&&) noexcept = default;
  IndexCacheHandle& operator=(IndexCacheHandle&&) noexcept = default;

  IndexRef index_ref() const { return ref_; }
  bool valid() const { return ref_ != nullptr; }

 private:
  IndexRef ref_;
  std::shared_ptr<void> releaser_;  // destructor triggers the impl's Release hook
};

// Global injection point. Last writer wins; intended to be called once during
// process init before any reader/searcher construction.
void SetGlobalIndexCache(IndexCache* cache);
IndexCache* GetGlobalIndexCache();
}  // namespace tenann
