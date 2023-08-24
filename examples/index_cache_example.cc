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

#include <iostream>

#include "tenann/common/logging.h"
#include "tenann/index/index_cache.h"

using namespace tenann;

class IndexMock {
 public:
  IndexMock() : name() {}
  IndexMock(const std::string& name) : name(name){};
  ~IndexMock() { T_LOG(INFO) << "Index destroyed: " << name; }

  std::string name;
};

IndexRef CreateIndex(const std::string& name) {
  IndexRef index_ref =
      std::make_shared<Index>(new IndexMock(name),    //
                              IndexType::kFaissHnsw,  //
                              [](void* index) { delete reinterpret_cast<IndexMock*>(index); });
  return index_ref;
}

void ReadWriteExample() {
  IndexRef index_ref = CreateIndex("index1");

  T_LOG(INFO) << "Index built: " << reinterpret_cast<IndexMock*>(index_ref->index_raw())->name;

  // write index to cache
  IndexCacheHandle write_handle;
  auto* cache = IndexCache::GetGlobalInstance();
  cache->Insert("index1", index_ref, &write_handle);

  // read index from cache
  IndexCacheHandle read_handle;
  auto found = cache->Lookup("index1", &read_handle);
  T_CHECK(found);

  // There should be two references to the index: one is original index_ref, and another is held by
  // the cache.
  T_LOG(INFO) << "IndexRef use count: " << index_ref.use_count();

  auto shared_ref_from_cache = read_handle.index_ref();
  // There should be three references to the index:
  // 1. `index_ref`
  // 2. the reference held by the cache
  // 3. `shared_ref_from_cache`
  T_LOG(INFO) << "IndexRef use count: " << index_ref.use_count();

  T_LOG(INFO) << "Index read from cache: "
              << reinterpret_cast<const IndexMock*>(shared_ref_from_cache->index_raw())->name;
}

void EvictExample() {
  auto cache = IndexCache::GetGlobalInstance();
  cache->SetCapacity(2);

  auto index1 = CreateIndex("index1");
  auto index2 = CreateIndex("index2");
  auto index3 = CreateIndex("index3");

  // insert index1 to cache
  {
    IndexCacheHandle handle;
    cache->Insert("index1", std::move(index1), &handle);
  }

  // lookup index1 for 10 times
  for (int i = 0; i < 10; i++) {
    IndexCacheHandle handle;
    cache->Lookup("index1", &handle);
  }

  // insert index2 to cache
  {
    IndexCacheHandle handle;
    cache->Insert("index2", std::move(index2), &handle);
  }

  // Insert index3 into the cache and monitor which one (index1 or index2) will be evicted.
  {
    IndexCacheHandle handle;
    cache->Insert("index3", std::move(index3), &handle);
  }
}

int main() {
  EvictExample();
  return 0;
}
