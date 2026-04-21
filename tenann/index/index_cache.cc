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

#include "tenann/index/index_cache.h"

#include "tenann/common/logging.h"

namespace tenann {

namespace {

// Precondition: `cache` must outlive every IndexCacheHandle produced from it.
// The singleton `IndexCache::GetGlobalInstance()` satisfies this trivially
// (function-local static lives until process exit). For tests and examples
// that construct on-stack IndexCache instances, ensure all handles are
// released before the owning IndexCache leaves scope.
//
// Build a releaser that calls Cache::release when the IndexCacheHandle is
// destroyed. The shared_ptr<void>'s deleter owns the Cache::Handle* and
// drops the pinning reference acquired by Cache::lookup / Cache::insert.
std::shared_ptr<void> MakeReleaser(Cache* cache, Cache::Handle* handle) {
  return std::shared_ptr<void>(
      reinterpret_cast<void*>(handle),
      [cache](void* opaque) {
        if (opaque != nullptr) {
          cache->release(static_cast<Cache::Handle*>(opaque));
        }
      });
}

}  // namespace

IndexCache::IndexCache(size_t capacity) : cache_(new_lru_cache(capacity)) {}

IndexCache::~IndexCache() = default;

IndexCache* IndexCache::GetGlobalInstance() {
  // The default cache capacity is 1GB
  static IndexCache instance(1024 * 1024 * 1024);
  return &instance;
}

bool IndexCache::Lookup(const CacheKey& key, IndexCacheHandle* handle) {
  auto* lru_handle = cache_->lookup(key);
  if (lru_handle == nullptr) {
    return false;
  }
  // The cache stores a heap-allocated IndexRef* (see Insert). Dereference to
  // share ownership of the underlying tenann::Index with the caller.
  auto* stored_ref = reinterpret_cast<IndexRef*>(cache_->value(lru_handle));
  *handle = IndexCacheHandle(*stored_ref, MakeReleaser(cache_.get(), lru_handle));
  return true;
}

void IndexCache::Insert(const CacheKey& key, IndexRef index, IndexCacheHandle* handle) {
  size_t index_size = index->EstimateMemoryUsage();
  // create a new reference to the index and intentionally leak the reference
  void* leaked_index = reinterpret_cast<void*>(new IndexRef(index));

  // the reference will not be destroyed until we manually delete it through a custom deleter
  auto deleter = [](const CacheKey& key, void* value) {
    auto leaked_index = reinterpret_cast<IndexRef*>(value);
    delete leaked_index;
  };

  CachePriority priority = CachePriority::NORMAL;

  auto* lru_handle = cache_->insert(key, leaked_index, index_size, deleter, priority);
  *handle = IndexCacheHandle(index, MakeReleaser(cache_.get(), lru_handle));
}

bool IndexCache::GetOrCreate(const CacheKey& key, const IndexLoader& loader,
                             IndexCacheHandle* handle) {
  // Fast-path: already cached.
  if (Lookup(key, handle)) return true;

  // Serialize concurrent misses on the same key.
  auto load_lock = get_or_create_load_lock(key.to_string());
  std::lock_guard<std::mutex> l(*load_lock);

  // Another caller may have populated the entry while we were waiting.
  if (Lookup(key, handle)) return true;

  IndexRef ref = loader();
  if (ref == nullptr) {
    T_LOG(ERROR) << "IndexLoader returned null IndexRef for key " << key.to_string();
    return false;
  }
  Insert(key, std::move(ref), handle);
  return false;
}

std::shared_ptr<std::mutex> IndexCache::get_or_create_load_lock(const std::string& key) {
  std::lock_guard<std::mutex> l(load_locks_mu_);
  auto it = load_locks_.find(key);
  if (it != load_locks_.end()) {
    if (auto sp = it->second.lock()) return sp;
  }
  auto sp = std::make_shared<std::mutex>();
  load_locks_[key] = sp;
  // Opportunistically reap expired entries so the map doesn't grow unbounded.
  for (auto mit = load_locks_.begin(); mit != load_locks_.end();) {
    if (mit->second.expired()) {
      mit = load_locks_.erase(mit);
    } else {
      ++mit;
    }
  }
  return sp;
}

void IndexCache::SetCapacity(size_t capacity) { cache_->set_capacity(capacity); }

bool IndexCache::AdjustCapacity(int64_t delta, size_t min_capacity) {
  return cache_->adjust_capacity(delta, min_capacity);
}

json IndexCache::status() const {
  json doc;
  cache_->get_cache_status(&doc);
  return doc;
}

size_t IndexCache::memory_usage() const { return cache_->get_memory_usage(); }

size_t IndexCache::capacity() { return cache_->get_capacity(); }

uint64_t IndexCache::lookup_count() { return cache_->get_lookup_count(); }

uint64_t IndexCache::hit_count() { return cache_->get_hit_count(); }

}  // namespace tenann
