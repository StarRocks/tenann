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
 */

#include "tenann/index/index_cache.h"

#include <atomic>

namespace tenann {

namespace {
std::atomic<IndexCache*> g_index_cache{nullptr};
}  // namespace

void SetGlobalIndexCache(IndexCache* cache) {
  g_index_cache.store(cache, std::memory_order_release);
}

IndexCache* GetGlobalIndexCache() {
  return g_index_cache.load(std::memory_order_acquire);
}

}  // namespace tenann
