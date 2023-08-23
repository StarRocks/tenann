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

#include "tenann/store/index_cache.h"

#include "tenann/common/logging.h"

namespace tenann {

IndexCache::IndexCache(size_t capacity) : cache_(new_lru_cache(capacity)) {}

IndexCache::~IndexCache() = default;

IndexCache* IndexCache::GetGlobalInstance() {
  // The default cache can hold 1024 index files
  static IndexCache instance(1024);
  return &instance;
}

bool IndexCache::Lookup(const CacheKey& key, IndexCacheEntry* handle) {
  auto* lru_handle = cache_->lookup(key);
  if (lru_handle == nullptr) {
    return false;
  }
  *handle = IndexCacheEntry(cache_.get(), lru_handle);
  return true;
}

void IndexCache::Insert(const CacheKey& key, IndexRef index, IndexCacheEntry* handle) {
  auto index_size = index->EstimateMemorySizeInBytes();
  // Create a new reference to the index and intentionally leak the reference
  void* leaked_index = reinterpret_cast<void*>(new IndexRef(index));

  // The reference will not be destroyed until we manually delete it through a custom deleter
  auto deleter = [](const CacheKey& key, void* value) {
    auto leaked_index = reinterpret_cast<IndexRef*>(value);
    delete leaked_index;
  };

  CachePriority priority = CachePriority::NORMAL;

  auto* lru_handle = cache_->insert(key, leaked_index, index_size, deleter, priority);
  *handle = IndexCacheEntry(cache_.get(), lru_handle);
}

size_t IndexCache::memory_usage() const { return cache_->get_memory_usage(); }

void IndexCache::SetCapacity(size_t capacity) { cache_->set_capacity(capacity); }

size_t IndexCache::capacity() { return cache_->get_capacity(); }

uint64_t IndexCache::lookup_count() { return cache_->get_lookup_count(); }

uint64_t IndexCache::hit_count() { return cache_->get_hit_count(); }

bool IndexCache::AdjustCapacity(int64_t delta, size_t min_capacity) {
  return cache_->adjust_capacity(delta, min_capacity);
}

IndexCacheEntry::IndexCacheEntry() = default;

IndexCacheEntry::IndexCacheEntry(Cache* cache, Cache::Handle* handle)
    : cache_(cache), handle_(handle) {}

IndexCacheEntry::~IndexCacheEntry() {
  if (handle_ != nullptr) {
    cache_->release(handle_);
  }
}

IndexCacheEntry::IndexCacheEntry(IndexCacheEntry&& other) noexcept {
  // we can use std::exchange if we switch c++14 on
  std::swap(cache_, other.cache_);
  std::swap(handle_, other.handle_);
}

IndexCacheEntry& IndexCacheEntry::operator=(IndexCacheEntry&& other) noexcept {
  std::swap(cache_, other.cache_);
  std::swap(handle_, other.handle_);
  return *this;
}

Cache* IndexCacheEntry::cache() const { return cache_; }

const Index* IndexCacheEntry::index() const {
  TNN_DCHECK(handle_ != nullptr);
  auto* handle = reinterpret_cast<LRUHandle*>(handle_);
  // The value saved in cache is a pointer to CompiledFunction
  auto* index = reinterpret_cast<IndexRef*>(handle->value)->get();
  return index;
}

}  // namespace tenann