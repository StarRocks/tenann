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

#include "tenann/index/index.h"

#include <gtest/gtest.h>

#include <cstdlib>

namespace tenann {

TEST(IndexExplicitBytes, ReturnsExplicitBytesWhenSet) {
    void* buf = std::malloc(128);
    Index idx(buf, IndexType::kFaissIvfPq,
              [](void* v) { std::free(v); },
              /*explicit_bytes=*/128);
    EXPECT_EQ(128u, idx.EstimateMemoryUsage());
}

TEST(IndexExplicitBytes, UnsupportedTypeReturnsOne) {
    // Use an IndexType that the type-based heuristic does NOT handle
    // (kFaissHnsw / kFaissIvfPq are the only handled cases), so we
    // exercise only the fallback path without constructing a real
    // faiss::Index and without triggering a static_cast on garbage
    // memory.
    Index idx(nullptr, IndexType::kFaissIvfPqOneInvertedList,
              [](void*) {});
    EXPECT_EQ(1u, idx.EstimateMemoryUsage());
}

}  // namespace tenann
