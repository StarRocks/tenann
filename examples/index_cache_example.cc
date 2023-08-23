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
#include "tenann/store/index_cache.h"

class IndexMock {
 public:
  IndexMock() : name() {}
  IndexMock(const std::string& name) : name(name){};
  ~IndexMock() { TNN_LOG(INFO) << "Index destroyed: " << name; }

  std::string name;
};

int main() {
  using namespace tenann;
  IndexRef index_ref =
      std::make_shared<Index>(new IndexMock("index1"),  //
                              IndexType::kFaissHnsw,    //
                              [](void* index) { delete reinterpret_cast<IndexMock*>(index); });

  TNN_LOG(INFO) << "index built: " << reinterpret_cast<IndexMock*>(index_ref->index_raw())->name;

  // write index to cache
  auto* cache = IndexCache::GetGlobalInstance();
  IndexCacheEntry write_entry;
  cache->Insert("index1", index_ref, &write_entry);

  // read index from cache
  IndexCacheEntry read_entry;
  auto found = cache->Lookup("index1", &read_entry);
  TNN_CHECK(found);

  // There should be two references to the index: one is index_ref, and another is held by the cache.
  TNN_LOG(INFO) << "index ref count: " << index_ref.use_count();
  TNN_LOG(INFO) << "index read from cache: "
                << reinterpret_cast<const IndexMock*>(read_entry.index()->index_raw())->name;
  return 0;
}