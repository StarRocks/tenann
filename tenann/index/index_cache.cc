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

#include "index_cache.h"
#include "tenann/common/logging.h"

namespace tenann {

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
  *handle = IndexCacheHandle(cache_.get(), lru_handle);
  return true;
}

void IndexCache::Insert(const CacheKey& key, IndexRef index, IndexCacheHandle* handle) {
  auto index_size = index->EstimateMemoryUsage();
  // create a new reference to the index and intentionally leak the reference
  void* leaked_index = reinterpret_cast<void*>(new IndexRef(index));

  // the reference will not be destroyed until we manually delete it through a custom deleter
  auto deleter = [](const CacheKey& key, void* value) {
    auto leaked_index = reinterpret_cast<IndexRef*>(value);
    delete leaked_index;
  };

  CachePriority priority = CachePriority::NORMAL;

  auto* lru_handle = cache_->insert(key, leaked_index, index_size, deleter, priority);
  *handle = IndexCacheHandle(cache_.get(), lru_handle);
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

IndexCacheHandle::IndexCacheHandle() = default;

IndexCacheHandle::IndexCacheHandle(Cache* cache, Cache::Handle* handle)
    : cache_(cache), handle_(handle) {}

IndexCacheHandle::~IndexCacheHandle() {
  if (handle_ != nullptr) {
    cache_->release(handle_);
  }
}

IndexCacheHandle::IndexCacheHandle(IndexCacheHandle&& other) noexcept {
  // we can use std::exchange if we switch c++14 on
  std::swap(cache_, other.cache_);
  std::swap(handle_, other.handle_);
}

IndexCacheHandle& IndexCacheHandle::operator=(IndexCacheHandle&& other) noexcept {
  std::swap(cache_, other.cache_);
  std::swap(handle_, other.handle_);
  return *this;
}

uint32_t IndexCacheHandle::cache_entry_ref_count() {
  T_DCHECK(handle_ != nullptr);
  auto* handle = reinterpret_cast<LRUHandle*>(handle_);
  return handle->refs;
}

Cache* IndexCacheHandle::cache() const { return cache_; }

IndexRef IndexCacheHandle::index_ref() const {
  T_DCHECK(handle_ != nullptr);
  auto* handle = reinterpret_cast<LRUHandle*>(handle_);
  // Ownership of the cache entry can be safely shared.
  // The index will be released if both the following conditions are satisfied:
  // 1. The cache entry is evicted from the cache.
  // 2. The reference count of the IndexRef instance becomes 0.
  auto shared_ref = *reinterpret_cast<IndexRef*>(handle->value);
  return shared_ref;
}

}  // namespace tenann