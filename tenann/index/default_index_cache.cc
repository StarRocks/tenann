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

#include "tenann/index/default_index_cache.h"

#include "tenann/common/logging.h"

namespace tenann {

namespace {

// Build a shared_ptr<void> that calls Cache::release when destroyed, so
// IndexCacheHandle unpins its entry via RAII. Precondition: `cache` outlives
// every handle produced from it.
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

DefaultIndexCache::DefaultIndexCache(size_t capacity) : cache_(new_lru_cache(capacity)) {}

DefaultIndexCache::~DefaultIndexCache() = default;

DefaultIndexCache* DefaultIndexCache::GetGlobalInstance() {
  static DefaultIndexCache instance(1024 * 1024 * 1024);  // 1 GiB
  return &instance;
}

bool DefaultIndexCache::Lookup(const CacheKey& key, IndexCacheHandle* handle) {
  auto* lru_handle = cache_->lookup(key);
  if (lru_handle == nullptr) {
    return false;
  }
  auto* stored_ref = reinterpret_cast<IndexRef*>(cache_->value(lru_handle));
  *handle = IndexCacheHandle(*stored_ref, MakeReleaser(cache_.get(), lru_handle));
  return true;
}

void DefaultIndexCache::Insert(const CacheKey& key, IndexRef index, IndexCacheHandle* handle) {
  size_t index_size = index->EstimateMemoryUsage();
  // Cache stores a heap-allocated IndexRef so ownership survives lookup;
  // the custom deleter drops the last reference on eviction.
  void* leaked_index = reinterpret_cast<void*>(new IndexRef(index));
  auto deleter = [](const CacheKey& key, void* value) {
    delete reinterpret_cast<IndexRef*>(value);
  };
  auto* lru_handle = cache_->insert(key, leaked_index, index_size, deleter, CachePriority::NORMAL);
  *handle = IndexCacheHandle(index, MakeReleaser(cache_.get(), lru_handle));
}

bool DefaultIndexCache::GetOrCreate(const CacheKey& key, const IndexLoader& loader,
                                    IndexCacheHandle* handle) {
  if (Lookup(key, handle)) return true;
  IndexRef ref = loader();
  if (ref == nullptr) {
    T_LOG(ERROR) << "IndexLoader returned null IndexRef for key " << key.to_string();
    *handle = IndexCacheHandle();
    return false;
  }
  Insert(key, std::move(ref), handle);
  return false;
}

void DefaultIndexCache::SetCapacity(size_t capacity) { cache_->set_capacity(capacity); }

bool DefaultIndexCache::AdjustCapacity(int64_t delta, size_t min_capacity) {
  return cache_->adjust_capacity(delta, min_capacity);
}

json DefaultIndexCache::status() const {
  json doc;
  cache_->get_cache_status(&doc);
  return doc;
}

size_t DefaultIndexCache::memory_usage() const { return cache_->get_memory_usage(); }

size_t DefaultIndexCache::capacity() { return cache_->get_capacity(); }

uint64_t DefaultIndexCache::lookup_count() { return cache_->get_lookup_count(); }

uint64_t DefaultIndexCache::hit_count() { return cache_->get_hit_count(); }

}  // namespace tenann
