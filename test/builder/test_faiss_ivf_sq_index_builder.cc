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

#include "faiss/Index.h"
#include "faiss/IndexScalarQuantizer.h"
#include "gtest/gtest.h"
#include "tenann/builder/faiss_ivf_sq_index_builder.h"
#include "tenann/common/seq_view.h"
#include "tenann/factory/index_factory.h"
#include "tenann/index/internal/faiss_index_util.h"
#include "tenann/store/index_meta.h"

namespace tenann {

namespace {

constexpr uint32_t kDim = 16;
constexpr uint32_t kNumRows = 1000;

IndexMeta MakeIvfSqMeta(size_t nbits) {
  IndexMeta meta;
  meta.SetMetaVersion(0);
  meta.SetIndexFamily(IndexFamily::kVectorIndex);
  meta.SetIndexType(IndexType::kFaissIvfSq);
  meta.common_params()["dim"] = static_cast<int>(kDim);
  meta.common_params()["metric_type"] = MetricType::kL2Distance;
  meta.index_params()["nlist"] = 16;
  meta.index_params()["nbits"] = nbits;
  meta.search_params()["nprobe"] = 16;
  meta.search_params()["max_codes"] = 0;
  return meta;
}

std::vector<float> RandomVectors() {
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
  std::vector<float> vectors(static_cast<size_t>(kNumRows) * kDim);
  for (auto& value : vectors) {
    value = distribution(rng);
  }
  return vectors;
}

void VerifyBuild(size_t nbits, faiss::ScalarQuantizer::QuantizerType expected_type) {
  auto meta = MakeIvfSqMeta(nbits);
  auto builder = IndexFactory::CreateBuilderFromMeta(meta);
  EXPECT_NE(dynamic_cast<FaissIvfSqIndexBuilder*>(builder.get()), nullptr);

  builder->Open();
  auto index_ref = builder->index_ref();
  ASSERT_NE(index_ref, nullptr);
  EXPECT_EQ(index_ref->index_type(), IndexType::kFaissIvfSq);

  auto* faiss_index = static_cast<faiss::Index*>(index_ref->index_raw());
  auto [transform, ivf_sq] = faiss_util::CheckAndUnpackIvfSq(faiss_index, nullptr);
  EXPECT_EQ(transform, nullptr);
  ASSERT_NE(ivf_sq, nullptr);
  EXPECT_EQ(ivf_sq->sq.qtype, expected_type);
  EXPECT_FALSE(ivf_sq->is_trained);

  auto vectors = RandomVectors();
  ArraySeqView base_view{.data = reinterpret_cast<uint8_t*>(vectors.data()),
                         .dim = kDim,
                         .size = kNumRows,
                         .elem_type = PrimitiveType::kFloatType};
  builder->Add({base_view}).Flush();
  EXPECT_TRUE(faiss_index->is_trained);
  EXPECT_EQ(faiss_index->ntotal, kNumRows);
  EXPECT_GT(index_ref->EstimateMemoryUsage(), 1u);
  builder->Close();
}

}  // namespace

TEST(FaissIvfSqIndexBuilderTest, BuildSq8) { VerifyBuild(8, faiss::ScalarQuantizer::QT_8bit); }

TEST(FaissIvfSqIndexBuilderTest, BuildSq4) { VerifyBuild(4, faiss::ScalarQuantizer::QT_4bit); }

TEST(FaissIvfSqIndexBuilderTest, SupportsCustomRowIdsAndNulls) {
  auto meta = MakeIvfSqMeta(8);
  auto vectors = RandomVectors();
  std::vector<int64_t> ids(kNumRows);
  std::vector<uint8_t> null_flags(kNumRows, 0);
  for (uint32_t i = 0; i < kNumRows; ++i) {
    ids[i] = 10000 + i;
    null_flags[i] = i % 5 == 0;
  }
  ArraySeqView base_view{.data = reinterpret_cast<uint8_t*>(vectors.data()),
                         .dim = kDim,
                         .size = kNumRows,
                         .elem_type = PrimitiveType::kFloatType};

  auto builder = IndexFactory::CreateBuilderFromMeta(meta);
  builder->EnableCustomRowId().Open().Add({base_view}, ids.data(), null_flags.data()).Flush();
  auto* faiss_index = static_cast<faiss::Index*>(builder->index_ref()->index_raw());
  EXPECT_EQ(faiss_index->ntotal, kNumRows - kNumRows / 5);
  builder->Close();
}

}  // namespace tenann
