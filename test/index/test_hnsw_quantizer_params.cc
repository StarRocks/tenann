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

#include "gtest/gtest.h"
#include "tenann/index/parameter_serde.h"
#include "tenann/index/parameters.h"
#include "tenann/store/index_meta.h"
#include "tenann/store/index_type.h"

namespace tenann {

TEST(HnswQuantizerParamsTest, DefaultValues) {
  FaissHnswIndexParams p;
  EXPECT_EQ(p.M, 16);
  EXPECT_EQ(p.efConstruction, 40);
  // New quantizer-related fields default to Flat.
  EXPECT_EQ(p.quantizer, static_cast<int>(ScalarQuantizerType::kFlat));
  EXPECT_EQ(p.m_pq, 0);
  EXPECT_EQ(p.nbits_pq, 8);
}

TEST(HnswQuantizerParamsTest, ValidateFlatDefault) {
  FaissHnswIndexParams p;
  // Default (Flat) should validate without errors or log fatal.
  p.Validate();
}

TEST(HnswQuantizerParamsTest, ValidateSq8) {
  FaissHnswIndexParams p;
  p.quantizer = static_cast<int>(ScalarQuantizerType::kSQ8);
  // SQ does not require m_pq/nbits_pq; Validate() should pass.
  p.Validate();
}

TEST(HnswQuantizerParamsTest, ValidatePqHappyPath) {
  FaissHnswIndexParams p;
  p.quantizer = static_cast<int>(ScalarQuantizerType::kPQ);
  p.m_pq = 16;
  p.nbits_pq = 8;
  p.Validate();
}

TEST(HnswQuantizerParamsTest, FetchParamsRoundtrip) {
  // Build an IndexMeta with the new fields and verify FetchParameters reads them.
  IndexMeta meta;
  meta.SetIndexFamily(IndexFamily::kVectorIndex);
  meta.SetIndexType(IndexType::kFaissHnsw);
  meta.index_params()["M"] = 32;
  meta.index_params()["efConstruction"] = 100;
  meta.index_params()["quantizer"] = static_cast<int>(ScalarQuantizerType::kPQ);
  meta.index_params()["m_pq"] = 8;
  meta.index_params()["nbits_pq"] = 8;
  meta.common_params()["dim"] = 128;
  meta.common_params()["metric_type"] = MetricType::kL2Distance;

  FaissHnswIndexParams p;
  FetchParameters(meta, &p);
  EXPECT_EQ(p.M, 32);
  EXPECT_EQ(p.efConstruction, 100);
  EXPECT_EQ(p.quantizer, static_cast<int>(ScalarQuantizerType::kPQ));
  EXPECT_EQ(p.m_pq, 8);
  EXPECT_EQ(p.nbits_pq, 8);
}

TEST(HnswQuantizerParamsTest, FetchParamsDefaultsForOldIndex) {
  // An older IndexMeta without the new fields should fall back to defaults.
  IndexMeta meta;
  meta.SetIndexFamily(IndexFamily::kVectorIndex);
  meta.SetIndexType(IndexType::kFaissHnsw);
  meta.index_params()["M"] = 16;
  meta.index_params()["efConstruction"] = 40;
  meta.common_params()["dim"] = 128;
  meta.common_params()["metric_type"] = MetricType::kL2Distance;

  FaissHnswIndexParams p;
  FetchParameters(meta, &p);
  EXPECT_EQ(p.quantizer, static_cast<int>(ScalarQuantizerType::kFlat));
  EXPECT_EQ(p.m_pq, 0);
  EXPECT_EQ(p.nbits_pq, 8);
}

}  // namespace tenann
