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
TEST(ShardedLRUCacheTest, InsertLookupAndRelease) {
  ShardedLRUCache cache(1000);
  Cache::Handle* handle;
  CacheKey key = "key1";
  std::string value = "value1";

  // 插入一个新的缓存条目
  handle = cache.insert(key, &value, value.size(), [](const CacheKey&, void*) {});
  ASSERT_NE(handle, nullptr);

  // 查找刚刚插入的缓存条目
  handle = cache.lookup(key);
  ASSERT_NE(handle, nullptr);
  EXPECT_EQ(*static_cast<std::string*>(cache.value(handle)), value);

  // 释放缓存条目
  cache.release(handle);
}

TEST(ShardedLRUCacheTest, Erase) {
  ShardedLRUCache cache(1000);
  Cache::Handle* handle;
  CacheKey key = "key1";
  std::string value = "value1";

  // 插入一个新的缓存条目
  handle = cache.insert(key, &value, value.size(), [](const CacheKey&, void*) {});
  ASSERT_NE(handle, nullptr);

  // 擦除刚刚插入的缓存条目
  cache.erase(key);

  // 查找擦除的缓存条目应该返回 nullptr
  handle = cache.lookup(key);
  EXPECT_EQ(handle, nullptr);
}

TEST(ShardedLRUCacheTest, NewId) {
  ShardedLRUCache cache(1000);
  uint64_t id1 = cache.new_id();
  uint64_t id2 = cache.new_id();

  // 新生成的 ID 应该是递增的
  EXPECT_LT(id1, id2);
}

TEST(ShardedLRUCacheTest, AdjustCapacity) {
  ShardedLRUCache cache(1000);
  CacheKey key = "key1";
  std::string value = "value1";
  Cache::Handle* handle;

  // 插入一个新的缓存条目
  handle = cache.insert(key, &value, value.size(), [](const CacheKey&, void*) {});
  ASSERT_NE(handle, nullptr);

  // 增加缓存容量
  cache.adjust_capacity(500);
  EXPECT_EQ(cache.get_capacity(), 1500);

  // 减少缓存容量
  cache.adjust_capacity(-500, 500);
  EXPECT_EQ(cache.get_capacity(), 1000);
}

TEST(ShardedLRUCacheTest, CapacityOverflow) {
  ShardedLRUCache cache(100);
  CacheKey key1 = "key1";
  std::string value1 = "value1";
  CacheKey key2 = "key2";
  std::string value2 = "value2";
  Cache::Handle* handle1;
  Cache::Handle* handle2;

  // 插入一个新的缓存条目，使缓存容量达到最大
  handle1 = cache.insert(key1, &value1, value1.size(), [](const CacheKey&, void*) {});
  ASSERT_NE(handle1, nullptr);
  // TODO: fix
  // EXPECT_EQ(cache.get_memory_usage(), 100);

  // 再插入一个新的缓存条目，此时缓存容量已满，应该会淘汰一些旧的缓存条目
  handle2 = cache.insert(key2, &value2, value2.size(), [](const CacheKey&, void*) {});
  ASSERT_NE(handle2, nullptr);
  // TODO: fix
  // EXPECT_EQ(cache.get_memory_usage(), 100);

  // 释放缓存条目
  cache.release(handle1);
  cache.release(handle2);
}
}  // namespace tenann