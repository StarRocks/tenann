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

#include "tenann/index/parameters.h"
#include "tenann/store/lru_cache.h"
#include "test/faiss_test_base.h"

namespace tenann {
class ShardedLRUCacheTest : public FaissTestBase {
 protected:
  void SetUp() override {
    cache_ = new tenann::ShardedLRUCache(1000);
  }

  void TearDown() override {
    delete cache_;
  }

  tenann::ShardedLRUCache* cache_;
};

TEST_F(ShardedLRUCacheTest, InsertLookupRelease) {
  // Test inserting, looking up, and releasing a cache entry.
  auto key = tenann::CacheKey{std::string("test_key")};
  auto value = new int(42);
  auto deleter = [](const tenann::CacheKey& key, void* value) {
    delete static_cast<int*>(value);
  };
  auto handle = cache_->insert(key, value, sizeof(int), deleter);
  EXPECT_NE(handle, nullptr);

  auto lookup_handle = cache_->lookup(key);
  EXPECT_EQ(lookup_handle, handle);
  EXPECT_EQ(*static_cast<int*>(cache_->value(handle)), 42);

  cache_->release(handle);
}

TEST_F(ShardedLRUCacheTest, Erase) {
  // Test inserting and erasing a cache entry.
  auto key = tenann::CacheKey{std::string("test_key")};
  auto value = new int(42);
  auto deleter = [](const tenann::CacheKey& key, void* value) {
    delete static_cast<int*>(value);
  };
  auto handle = cache_->insert(key, value, sizeof(int), deleter);
  EXPECT_NE(handle, nullptr);

  cache_->erase(key);
  auto lookup_handle = cache_->lookup(key);
  EXPECT_EQ(lookup_handle, nullptr);
}

TEST_F(ShardedLRUCacheTest, NewId) {
  // Test getting a new ID.
  auto id1 = cache_->new_id();
  auto id2 = cache_->new_id();
  EXPECT_NE(id1, id2);
}

TEST_F(ShardedLRUCacheTest, Prune) {
  // Test pruning the cache.
  auto key1 = tenann::CacheKey{std::string("test_key1")};
  auto value1 = new int(42);
  auto deleter1 = [](const tenann::CacheKey& key, void* value) {
    delete static_cast<int*>(value);
  };
  auto handle1 = cache_->insert(key1, value1, sizeof(int), deleter1);

  auto key2 = tenann::CacheKey{std::string("test_key2")};
  auto value2 = new int(43);
  auto deleter2 = [](const tenann::CacheKey& key, void* value) {
    delete static_cast<int*>(value);
  };
  auto handle2 = cache_->insert(key2, value2, sizeof(int), deleter2);

  cache_->prune();
  EXPECT_NE(cache_->lookup(key1), nullptr);
  EXPECT_NE(cache_->lookup(key2), nullptr);
}

TEST_F(ShardedLRUCacheTest, GetCacheStatus) {
  // Test getting the cache status.
  auto key = tenann::CacheKey{std::string("test_key")};
  auto value = new int(42);
  auto deleter = [](const tenann::CacheKey& key, void* value) {
    delete static_cast<int*>(value);
  };
  auto handle = cache_->insert(key, value, sizeof(int), deleter);

  json document;
  cache_->get_cache_status(&document);
  EXPECT_EQ(document[0]["capacity"], 500);
  EXPECT_EQ(document[0]["usage"], sizeof(int));
  EXPECT_EQ(document[0]["lookup_count"], 0);
  EXPECT_EQ(document[0]["hit_count"], 0);
}

TEST_F(ShardedLRUCacheTest, SetCapacity) {
  // Test setting the cache capacity.
  auto key = tenann::CacheKey{std::string("test_key")};
  auto value = new int(42);
  auto deleter = [](const tenann::CacheKey& key, void* value) {
    delete static_cast<int*>(value);
  };
  auto handle = cache_->insert(key, value, sizeof(int), deleter);

  cache_->set_capacity(2000);
  EXPECT_EQ(cache_->get_capacity(), 2000);
}

TEST_F(ShardedLRUCacheTest, GetLookupCount) {
  // Test getting the lookup count.
  auto key = tenann::CacheKey{std::string("test_key")};
  auto value = new int(42);
  auto deleter = [](const tenann::CacheKey& key, void* value) {
    delete static_cast<int*>(value);
  };
  auto handle = cache_->insert(key, value, sizeof(int), deleter);

  EXPECT_EQ(cache_->get_memory_usage(), sizeof(int));

  EXPECT_EQ(cache_->get_capacity(), 1000);
  EXPECT_TRUE(cache_->adjust_capacity(500));
  EXPECT_EQ(cache_->get_capacity(), 1500);

  cache_->lookup(key);
  EXPECT_EQ(cache_->get_lookup_count(), 1);
  EXPECT_EQ(cache_->get_hit_count(), 1);
}
}  // namespace tenann