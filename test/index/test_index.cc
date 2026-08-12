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
#include <memory>
#include <random>
#include <vector>

#include "faiss/IndexHNSW.h"
#include "faiss/index_factory.h"

namespace tenann {

namespace {

constexpr uint32_t kDim = 64;
// >= 39 * 256 so PQ8 training never falls below faiss' min-points-per-centroid
// (a smaller set only prints a clustering warning, but keeps test output noisy).
constexpr uint32_t kNumRows = 10000;

std::vector<float> RandomVectors(uint32_t n, uint32_t dim) {
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> v(static_cast<size_t>(n) * dim);
  for (auto& x : v) x = dist(rng);
  return v;
}

// Builds a real HNSW index with the given storage quantizer and returns the
// tenann::Index wrapper the cache charges through.
std::unique_ptr<Index> BuildHnsw(const char* factory_string, const std::vector<float>& data) {
  auto* raw = faiss::index_factory(kDim, factory_string, faiss::METRIC_L2);
  raw->train(kNumRows, data.data());
  raw->add(kNumRows, data.data());
  return std::make_unique<Index>(raw, IndexType::kFaissHnsw,
                                 [](void* p) { delete static_cast<faiss::Index*>(p); });
}

}  // namespace

TEST(IndexExplicitBytes, ReturnsExplicitBytesWhenSet) {
  void* buf = std::malloc(128);
  Index idx(buf, IndexType::kFaissIvfPq,
            [](void* v) { std::free(v); },
            /*explicit_bytes=*/128);
  EXPECT_EQ(128u, idx.EstimateMemoryUsage());
}

TEST(IndexExplicitBytes, UnsupportedTypeReturnsOne) {
  // Use an IndexType the heuristic does NOT handle (kFaissHnsw/kFaissIvfPq are
  // the only handled cases) so we exercise the fallback without constructing a
  // real faiss::Index or triggering a static_cast on garbage memory.
  Index idx(nullptr, IndexType::kFaissIvfPqOneInvertedList, [](void*) {});
  EXPECT_EQ(1u, idx.EstimateMemoryUsage());
}

// A quantized HNSW stores compressed codes, not fp32 vectors, so its charge must
// drop by exactly the bytes the quantizer saves. Charging fp32 unconditionally
// inflates the cache charge 4x (SQ8) to 32x (PQ), which shrinks the effective
// cache capacity and makes the vector_index memory metrics unusable.
TEST(IndexMemoryUsage, ChargesQuantizedHnswByStorageCodeSize) {
  const auto data = RandomVectors(kNumRows, kDim);
  const size_t vectors_fp32 = static_cast<size_t>(kNumRows) * kDim * sizeof(float);

  auto flat = BuildHnsw("HNSW8,Flat", data);
  auto sq8 = BuildHnsw("HNSW8,SQ8", data);
  auto pq8 = BuildHnsw("HNSW8,PQ8np", data);

  const size_t flat_usage = flat->EstimateMemoryUsage();
  const size_t sq8_usage = sq8->EstimateMemoryUsage();
  const size_t pq8_usage = pq8->EstimateMemoryUsage();

  // SQ8 keeps 1 byte per dimension instead of 4, so it saves 3/4 of the vector bytes.
  // The graph is identical across the three (same data, same M => same level
  // assignment), so the whole delta is the storage term.
  EXPECT_LT(sq8_usage, flat_usage);
  EXPECT_EQ(flat_usage - sq8_usage, vectors_fp32 / 4 * 3);

  // PQ8 keeps 8 bytes per vector regardless of dim, plus one shared codebook and the
  // symmetric distance table HNSW uses while constructing and maintaining the graph.
  const size_t pq_codebook = 256 * static_cast<size_t>(kDim) * sizeof(float);
  const size_t pq_sdc = 8 * 256 * 256 * sizeof(float);
  EXPECT_LT(pq8_usage, flat_usage);
  EXPECT_GT(pq8_usage, sq8_usage);
  EXPECT_EQ(flat_usage - pq8_usage,
            vectors_fp32 - static_cast<size_t>(kNumRows) * 8 - pq_codebook - pq_sdc);
}

// Guards the other direction: the fix must not undercharge a non-quantized index,
// whose storage really is fp32.
TEST(IndexMemoryUsage, StillChargesFlatHnswAsFp32) {
  const auto data = RandomVectors(kNumRows, kDim);
  auto flat = BuildHnsw("HNSW8,Flat", data);
  EXPECT_GE(flat->EstimateMemoryUsage(), static_cast<size_t>(kNumRows) * kDim * sizeof(float));
}

}  // namespace tenann
