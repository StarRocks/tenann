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

#include <memory>
#include <random>
#include <vector>

#include "gtest/gtest.h"
#include "tenann/builder/faiss_hnsw_index_builder.h"
#include "tenann/common/seq_view.h"
#include "tenann/factory/ann_searcher_factory.h"
#include "tenann/searcher/faiss_hnsw_ann_searcher.h"
#include "tenann/store/index_meta.h"
#include "tenann/store/index_type.h"

namespace tenann {

namespace {

constexpr uint32_t kDim = 32;
constexpr uint32_t kNb = 2000;
constexpr uint32_t kNq = 5;
constexpr uint32_t kK = 10;

IndexMeta MakeHnswMeta(ScalarQuantizerType q, int m_pq = 0, int nbits_pq = 8) {
  IndexMeta meta;
  meta.SetMetaVersion(0);
  meta.SetIndexFamily(IndexFamily::kVectorIndex);
  meta.SetIndexType(IndexType::kFaissHnsw);
  meta.common_params()["dim"] = static_cast<int>(kDim);
  meta.common_params()["is_vector_normed"] = false;
  meta.common_params()["metric_type"] = MetricType::kL2Distance;
  meta.index_params()["M"] = 16;
  meta.index_params()["efConstruction"] = 40;
  meta.index_params()["quantizer"] = static_cast<int>(q);
  if (q == ScalarQuantizerType::kPQ) {
    meta.index_params()["m_pq"] = m_pq;
    meta.index_params()["nbits_pq"] = nbits_pq;
  }
  meta.search_params()["efSearch"] = 40;
  meta.search_params()["check_relative_distance"] = true;
  meta.index_writer_options()["write_index_cache"] = false;
  meta.index_reader_options()["cache_index_file"] = false;
  return meta;
}

std::vector<float> RandomVectors(uint32_t n, uint32_t dim, int seed = 42) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> v(static_cast<size_t>(n) * dim);
  for (auto& x : v) x = dist(rng);
  return v;
}

std::string MakeIndexPath(const std::string& tag) {
  return "/tmp/tenann_hnsw_quantized_test_" + tag + ".index";
}

}  // namespace

// ---------- GetMinTrainRows ----------

// GetMinTrainRows must work pre-Open() — callers (e.g. StarRocks async build)
// use it to decide whether to build or fall back to brute-force before any
// training work begins.
TEST(FaissHnswQuantizedBuilderTest, MinTrainRowsFlat) {
  auto meta = MakeHnswMeta(ScalarQuantizerType::kFlat);
  FaissHnswIndexBuilder b(meta);
  EXPECT_EQ(b.GetMinTrainRows(), 0u);
}

TEST(FaissHnswQuantizedBuilderTest, MinTrainRowsSq4) {
  auto meta = MakeHnswMeta(ScalarQuantizerType::kSQ4);
  FaissHnswIndexBuilder b(meta);
  EXPECT_EQ(b.GetMinTrainRows(), 1u);
}

TEST(FaissHnswQuantizedBuilderTest, MinTrainRowsSq8) {
  auto meta = MakeHnswMeta(ScalarQuantizerType::kSQ8);
  FaissHnswIndexBuilder b(meta);
  EXPECT_EQ(b.GetMinTrainRows(), 1u);
}

TEST(FaissHnswQuantizedBuilderTest, MinTrainRowsPqDefaultNbits) {
  // nbits_pq=8 -> 256 centroids -> 25600 rows
  auto meta = MakeHnswMeta(ScalarQuantizerType::kPQ, /*m_pq=*/8, /*nbits_pq=*/8);
  FaissHnswIndexBuilder b(meta);
  EXPECT_EQ(b.GetMinTrainRows(), (1u << 8) * 100u);
}

TEST(FaissHnswQuantizedBuilderTest, MinTrainRowsPqSmallNbits) {
  // nbits_pq=4 -> 16 centroids -> 1600 rows
  auto meta = MakeHnswMeta(ScalarQuantizerType::kPQ, /*m_pq=*/8, /*nbits_pq=*/4);
  FaissHnswIndexBuilder b(meta);
  EXPECT_EQ(b.GetMinTrainRows(), (1u << 4) * 100u);
}

// ---------- Build + flush + search round-trip ----------

namespace {

void RunBuildAndSearch(const IndexMeta& meta, const std::string& path_tag,
                       float min_self_recall) {
  auto base = RandomVectors(kNb, kDim, /*seed=*/1);
  std::vector<int64_t> ids(kNb);
  for (uint32_t i = 0; i < kNb; ++i) ids[i] = i;

  ArraySeqView base_view{.data = reinterpret_cast<uint8_t*>(base.data()),
                         .dim = kDim,
                         .size = kNb,
                         .elem_type = PrimitiveType::kFloatType};

  auto path = MakeIndexPath(path_tag);
  auto builder = std::make_unique<FaissHnswIndexBuilder>(meta);
  builder->EnableCustomRowId()
      .Open(path)
      .Add({base_view}, ids.data())
      .Flush()
      .Close();

  // Self-query: pick the first kNq base vectors as queries; the nearest
  // neighbour should be the vector itself for any reasonable quantizer.
  auto searcher = AnnSearcherFactory::CreateSearcherFromMeta(meta);
  searcher->ReadIndex(path);

  std::vector<int64_t> result(kK);
  uint32_t hits = 0;
  for (uint32_t i = 0; i < kNq; ++i) {
    PrimitiveSeqView q{.data = reinterpret_cast<uint8_t*>(base.data() + i * kDim),
                       .size = kDim,
                       .elem_type = PrimitiveType::kFloatType};
    std::fill(result.begin(), result.end(), -1);
    searcher->AnnSearch(q, kK, result.data());
    for (int64_t r : result) {
      if (r == static_cast<int64_t>(i)) {
        ++hits;
        break;
      }
    }
  }
  EXPECT_GE(static_cast<float>(hits) / kNq, min_self_recall)
      << "self-recall too low for tag=" << path_tag;
}

}  // namespace

TEST(FaissHnswQuantizedBuilderTest, BuildAndSearchFlat) {
  // Baseline: HNSWFlat (no quantization, train path bypassed). Self-recall
  // should be perfect.
  RunBuildAndSearch(MakeHnswMeta(ScalarQuantizerType::kFlat), "flat",
                    /*min_self_recall=*/1.0f);
}

TEST(FaissHnswQuantizedBuilderTest, BuildAndSearchSq8) {
  // SQ8 needs train-then-add. Quantization is fine-grained enough that
  // self-query still recovers the vector.
  RunBuildAndSearch(MakeHnswMeta(ScalarQuantizerType::kSQ8), "sq8",
                    /*min_self_recall=*/1.0f);
}

TEST(FaissHnswQuantizedBuilderTest, BuildAndSearchSq4) {
  // SQ4 is coarser; allow the occasional miss.
  RunBuildAndSearch(MakeHnswMeta(ScalarQuantizerType::kSQ4), "sq4",
                    /*min_self_recall=*/0.6f);
}

TEST(FaissHnswQuantizedBuilderTest, BuildAndSearchPq) {
  // dim=32, m_pq=4 (each subvector is 8 floats), nbits_pq=4 -> 16 centroids
  // -> 1600 minimum training rows; we have kNb=2000.
  RunBuildAndSearch(MakeHnswMeta(ScalarQuantizerType::kPQ, /*m_pq=*/4, /*nbits_pq=*/4),
                    "pq4x4", /*min_self_recall=*/0.6f);
}

}  // namespace tenann
