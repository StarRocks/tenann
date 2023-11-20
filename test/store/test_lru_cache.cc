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
// TEST(ShardedLRUCacheTest, InsertLookupAndRelease) {
//   ShardedLRUCache cache(1000);
//   Cache::Handle* handle;
//   CacheKey key = "key1";
//   std::string value = "value1";

//   // 插入一个新的缓存条目
//   handle = cache.insert(key, &value, value.size(), [](const CacheKey&, void*) {});
//   ASSERT_NE(handle, nullptr);

//   // 查找刚刚插入的缓存条目
//   handle = cache.lookup(key);
//   ASSERT_NE(handle, nullptr);
//   EXPECT_EQ(*static_cast<std::string*>(cache.value(handle)), value);

//   // 释放缓存条目
//   cache.release(handle);
// }

// TEST(ShardedLRUCacheTest, Erase) {
//   ShardedLRUCache cache(1000);
//   Cache::Handle* handle;
//   CacheKey key = "key1";
//   std::string value = "value1";

//   // 插入一个新的缓存条目
//   handle = cache.insert(key, &value, value.size(), [](const CacheKey&, void*) {});
//   ASSERT_NE(handle, nullptr);

//   // 擦除刚刚插入的缓存条目
//   cache.erase(key);

//   // 查找擦除的缓存条目应该返回 nullptr
//   handle = cache.lookup(key);
//   EXPECT_EQ(handle, nullptr);
// }

// TEST(ShardedLRUCacheTest, NewId) {
//   ShardedLRUCache cache(1000);
//   uint64_t id1 = cache.new_id();
//   uint64_t id2 = cache.new_id();

//   // 新生成的 ID 应该是递增的
//   EXPECT_LT(id1, id2);
// }

// TEST(ShardedLRUCacheTest, AdjustCapacity) {
//   ShardedLRUCache cache(1000);
//   CacheKey key = "key1";
//   std::string value = "value1";
//   Cache::Handle* handle;

//   // 插入一个新的缓存条目
//   handle = cache.insert(key, &value, value.size(), [](const CacheKey&, void*) {});
//   ASSERT_NE(handle, nullptr);

//   // 增加缓存容量
//   cache.adjust_capacity(500);
//   EXPECT_EQ(cache.get_capacity(), 1500);

//   // 减少缓存容量
//   cache.adjust_capacity(-500, 500);
//   EXPECT_EQ(cache.get_capacity(), 1000);
// }

// TEST(ShardedLRUCacheTest, CapacityOverflow) {
//   ShardedLRUCache cache(100);
//   CacheKey key1 = "key1";
//   std::string value1 = "value1";
//   CacheKey key2 = "key2";
//   std::string value2 = "value2";
//   Cache::Handle* handle1;
//   Cache::Handle* handle2;

//   // 插入一个新的缓存条目，使缓存容量达到最大
//   handle1 = cache.insert(key1, &value1, value1.size(), [](const CacheKey&, void*) {});
//   ASSERT_NE(handle1, nullptr);
//   // TODO: fix
//   // EXPECT_EQ(cache.get_memory_usage(), 100);

//   // 再插入一个新的缓存条目，此时缓存容量已满，应该会淘汰一些旧的缓存条目
//   handle2 = cache.insert(key2, &value2, value2.size(), [](const CacheKey&, void*) {});
//   ASSERT_NE(handle2, nullptr);
//   // TODO: fix
//   // EXPECT_EQ(cache.get_memory_usage(), 100);

//   // 释放缓存条目
//   cache.release(handle1);
//   cache.release(handle2);
// }

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

TEST_F(ShardedLRUCacheTest, GetMemoryUsage) {
  // Test getting the memory usage.
  auto key = tenann::CacheKey{std::string("test_key")};
  auto value = new int(42);
  auto deleter = [](const tenann::CacheKey& key, void* value) {
    delete static_cast<int*>(value);
  };
  auto handle = cache_->insert(key, value, sizeof(int), deleter);

  EXPECT_EQ(cache_->get_memory_usage(), sizeof(int));
}

TEST_F(ShardedLRUCacheTest, GetCapacity) {
  // Test getting the cache capacity.
  EXPECT_EQ(cache_->get_capacity(), 1000);
}

TEST_F(ShardedLRUCacheTest, GetLookupCount) {
  // Test getting the lookup count.
  auto key = tenann::CacheKey{std::string("test_key")};
  auto value = new int(42);
  auto deleter = [](const tenann::CacheKey& key, void* value) {
    delete static_cast<int*>(value);
  };
  auto handle = cache_->insert(key, value, sizeof(int), deleter);

  cache_->lookup(key);
  EXPECT_EQ(cache_->get_lookup_count(), 1);
}

TEST_F(ShardedLRUCacheTest, GetHitCount) {
  // Test getting the hit count.
  auto key = tenann::CacheKey{std::string("test_key")};
  auto value = new int(42);
  auto deleter = [](const tenann::CacheKey& key, void* value) {
    delete static_cast<int*>(value);
  };
  auto handle = cache_->insert(key, value, sizeof(int), deleter);

  cache_->lookup(key);
  EXPECT_EQ(cache_->get_hit_count(), 1);
}

TEST_F(ShardedLRUCacheTest, AdjustCapacity) {
  // Test adjusting the cache capacity.
  auto key = tenann::CacheKey{std::string("test_key")};
  auto value = new int(42);
  auto deleter = [](const tenann::CacheKey& key, void* value) {
    delete static_cast<int*>(value);
  };
  auto handle = cache_->insert(key, value, sizeof(int), deleter);

  EXPECT_TRUE(cache_->adjust_capacity(500));
  EXPECT_EQ(cache_->get_capacity(), 1500);
}
}  // namespace tenann