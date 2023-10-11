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
#include "tenann/util/bruteforce_ann.h"

class BruteForceAnnTest : public ::testing::Test {
 public:
  static constexpr const size_t dim = 2;    // vector dimension
  static constexpr const uint32_t nb = 10;  // number of base vectors
  static constexpr const uint32_t nq = 2;   // number of query vectors
  static constexpr const uint32_t k = 2;    // perform kNN search
 protected:
  void SetUp() override {
    base = {10, 20,  //
            20, 30,  //
            30, 40,  //
            40, 50,  //
            50, 60,  //
            45, 55,  //
            35, 45,  //
            25, 35,  //
            15, 25,  //
            5,  15};

    query = {5, 5,  //
             50, 50};

    base_view = tenann::ArraySeqView{.data = reinterpret_cast<uint8_t*>(base.data()),  //
                                     .dim = dim,                                       //
                                     .size = nb,                                       //
                                     .elem_type = tenann::PrimitiveType::kFloatType};

    query_view = tenann::ArraySeqView{.data = reinterpret_cast<uint8_t*>(query.data()),  //
                                      .dim = dim,                                        //
                                      .size = nq,                                        //
                                      .elem_type = tenann::PrimitiveType::kFloatType};

    actual_cosine_similarity = {
        0.01315397, 0.13153973,  // 0
        0.02192329, 0.21923289,  // 1
        0.0306926,  0.30692604,  // 2
        0.03946192, 0.3946192,   // 3
        0.04823124, 0.48231235,  // 4
        0.04384658, 0.43846577,  // 5
        0.03507726, 0.35077262,  // 6
        0.02630795, 0.26307946,  // 7
        0.01753863, 0.17538631,  // 8
        0.00876932, 0.08769315   // 9
    };

    actual_l2_distance = {
        250,  2500,  // 0
        850,  1300,  // 1
        1850, 500,   // 2
        3250, 100,   // 3
        5050, 100,   // 4
        4100, 50,    // 5
        2500, 250,   // 6
        1300, 850,   // 7
        500,  1850,  // 8
        100,  3250   // 9
    };
  }
  void TearDown() override {}

 protected:
  tenann::ArraySeqView base_view;
  tenann::ArraySeqView query_view;
  std::vector<float> base;
  std::vector<float> query;

  std::vector<float> actual_l2_distance;
  std::vector<float> actual_cosine_similarity;
};

TEST_F(BruteForceAnnTest, test_l2_distance_raw) {
  std::vector<float> result_distances(nq * k);
  std::vector<int64_t> result_ids(nq * k);
  tenann::util::BruteForceAnn(dim, base_view, nullptr, nullptr, query_view,
                              tenann::MetricType::kL2Distance, k, result_ids.data(),
                              result_distances.data());

  EXPECT_EQ(result_ids[0], 9);
  EXPECT_EQ(result_ids[1], 0);
  EXPECT_EQ(result_ids[2], 5);
  EXPECT_EQ(result_ids[3], 4);
}

TEST_F(BruteForceAnnTest, test_l2_distance_raw_with_rowid) {
  std::vector<float> result_distances(nq * k);
  std::vector<int64_t> result_ids(nq * k);
  std::vector<int64_t> base_ids(nb);
  for (int i = 0; i < nb; i++) {
    base_ids[i] = i + 1;
  }

  tenann::util::BruteForceAnn(dim, base_view, nullptr, base_ids.data(), query_view,
                              tenann::MetricType::kL2Distance, k, result_ids.data(),
                              result_distances.data());

  EXPECT_EQ(result_ids[0], 10);
  EXPECT_EQ(result_ids[1], 1);
  EXPECT_EQ(result_ids[2], 6);
  EXPECT_EQ(result_ids[3], 5);
}

TEST_F(BruteForceAnnTest, test_l2_distance_raw_with_nulls) {
  std::vector<float> result_distances(nq * k);
  std::vector<int64_t> result_ids(nq * k);

  std::vector<int64_t> base_ids(nb);
  for (int i = 0; i < nb; i++) {
    base_ids[i] = i;
  }

  std::vector<uint8_t> null_flags(nb);
  std::fill_n(null_flags.begin(), nb, 1);
  std::fill_n(null_flags.begin(), nb / 2, 0);

  tenann::util::BruteForceAnn(dim, base_view, null_flags.data(), base_ids.data(), query_view,
                              tenann::MetricType::kL2Distance, k, result_ids.data(),
                              result_distances.data());

  EXPECT_EQ(result_ids[0], 0);
  EXPECT_EQ(result_ids[1], 1);
  EXPECT_EQ(result_ids[2], 4);
  EXPECT_EQ(result_ids[3], 3);
}

TEST_F(BruteForceAnnTest, test_cosine_similarity_raw) {
  std::vector<float> result_distances(nq * k);
  std::vector<int64_t> result_ids(nq * k);
  tenann::util::BruteForceAnn(dim, base_view, nullptr, nullptr, query_view,
                              tenann::MetricType::kCosineSimilarity, k, result_ids.data(),
                              result_distances.data());

  EXPECT_EQ(result_ids[0], 4);
  EXPECT_EQ(result_ids[1], 5);
  EXPECT_EQ(result_ids[2], 4);
  EXPECT_EQ(result_ids[3], 5);
}
