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

#include <cstdint>
#include <limits>
#include <vector>

#include "gtest/gtest.h"
#include "tenann/util/distance_util.h"

TEST(DistanceUtilTest, test_reserve_topk_ascending) {
  std::vector<int64_t> ids = {0, 1, 2, 3, 4};
  std::vector<float> dis = {0, 3, 8, 1, 2};

  tenann::ReserveTopK(&ids, &dis, 3);

  EXPECT_EQ(3, ids.size());
  EXPECT_EQ(3, dis.size());

  // Check top-k ids
  EXPECT_EQ(ids[0], 0);
  EXPECT_EQ(ids[1], 3);
  EXPECT_EQ(ids[2], 4);

  // Chek top-k distances
  EXPECT_EQ(dis[0], 0);
  EXPECT_EQ(dis[1], 1);
  EXPECT_EQ(dis[2], 2);
}

TEST(DistanceUtilTest, test_reserve_topk_descending) {
  std::vector<int64_t> ids = {0, 1, 2, 3, 4};
  std::vector<float> dis = {0, 3, 8, 1, 2};

  tenann::ReserveTopK(&ids, &dis, 3, /*ascending=*/false);

  EXPECT_EQ(3, ids.size());
  EXPECT_EQ(3, dis.size());

  // Check top-k ids
  EXPECT_EQ(ids[0], 2);
  EXPECT_EQ(ids[1], 1);
  EXPECT_EQ(ids[2], 4);

  // Chek top-k distances
  EXPECT_EQ(dis[0], 8);
  EXPECT_EQ(dis[1], 3);
  EXPECT_EQ(dis[2], 2);
}

TEST(DistanceUtilTest, test_reserve_topk_int64_ids_ascending) {
  const int64_t id_bit31 = int64_t{1} << 31;
  const int64_t id_high = (int64_t{1} << 40) + 7;
  const int64_t id_max = std::numeric_limits<int64_t>::max();
  const int64_t id_neg_high = -(int64_t{1} << 40) - 9;
  const int64_t id_min = std::numeric_limits<int64_t>::min();

  std::vector<int64_t> ids = {id_bit31, id_high, id_max, id_neg_high, id_min, -1};
  std::vector<float> dis = {3.0f, 1.0f, 2.0f, 2.5f, 0.5f, 4.0f};

  tenann::ReserveTopK(&ids, &dis, 5);

  const std::vector<int64_t> expected_ids = {id_min, id_high, id_max, id_neg_high, id_bit31};
  const std::vector<float> expected_dis = {0.5f, 1.0f, 2.0f, 2.5f, 3.0f};
  EXPECT_EQ(ids, expected_ids);
  EXPECT_EQ(dis, expected_dis);
}

TEST(DistanceUtilTest, test_reserve_topk_int64_ids_descending) {
  const int64_t id_bit31 = int64_t{1} << 31;
  const int64_t id_high = (int64_t{1} << 40) + 7;
  const int64_t id_max = std::numeric_limits<int64_t>::max();
  const int64_t id_neg_high = -(int64_t{1} << 40) - 9;
  const int64_t id_min = std::numeric_limits<int64_t>::min();

  std::vector<int64_t> ids = {id_bit31, id_high, id_max, id_neg_high, id_min, -1};
  std::vector<float> dis = {3.0f, 1.0f, 2.0f, 2.5f, 0.5f, 4.0f};

  tenann::ReserveTopK(&ids, &dis, 5, /*ascending=*/false);

  const std::vector<int64_t> expected_ids = {-1, id_bit31, id_neg_high, id_max, id_high};
  const std::vector<float> expected_dis = {4.0f, 3.0f, 2.5f, 2.0f, 1.0f};
  EXPECT_EQ(ids, expected_ids);
  EXPECT_EQ(dis, expected_dis);
}

TEST(DistanceUtilTest, test_reserve_topk_zero_k) {
  std::vector<int64_t> ids_asc = {1, 2, 3};
  std::vector<float> dis_asc = {3.0f, 1.0f, 2.0f};
  tenann::ReserveTopK(&ids_asc, &dis_asc, 0);
  EXPECT_TRUE(ids_asc.empty());
  EXPECT_TRUE(dis_asc.empty());

  std::vector<int64_t> ids_desc = {1, 2, 3};
  std::vector<float> dis_desc = {3.0f, 1.0f, 2.0f};
  tenann::ReserveTopK(&ids_desc, &dis_desc, 0, /*ascending=*/false);
  EXPECT_TRUE(ids_desc.empty());
  EXPECT_TRUE(dis_desc.empty());
}

TEST(DistanceUtilTest, test_reserve_topk_negative_k_throws) {
  std::vector<int64_t> ids = {1, 2, 3};
  std::vector<float> dis = {3.0f, 1.0f, 2.0f};

  EXPECT_THROW(tenann::ReserveTopK(&ids, &dis, -1), tenann::Error);
  EXPECT_THROW(tenann::ReserveTopK(&ids, &dis, -1, /*ascending=*/false), tenann::Error);
  EXPECT_THROW(tenann::ReserveTopK(&ids, &dis, std::numeric_limits<int64_t>::min()), tenann::Error);
}

TEST(DistanceUtilTest, test_reserve_topk_k_greater_than_size) {
  std::vector<int64_t> ids_asc = {10, 20, 30};
  std::vector<float> dis_asc = {3.0f, 1.0f, 2.0f};
  tenann::ReserveTopK(&ids_asc, &dis_asc, 10);
  EXPECT_EQ(ids_asc, (std::vector<int64_t>{20, 30, 10}));
  EXPECT_EQ(dis_asc, (std::vector<float>{1.0f, 2.0f, 3.0f}));

  std::vector<int64_t> ids_desc = {10, 20, 30};
  std::vector<float> dis_desc = {3.0f, 1.0f, 2.0f};
  tenann::ReserveTopK(&ids_desc, &dis_desc, 10, /*ascending=*/false);
  EXPECT_EQ(ids_desc, (std::vector<int64_t>{10, 30, 20}));
  EXPECT_EQ(dis_desc, (std::vector<float>{3.0f, 2.0f, 1.0f}));

  std::vector<int64_t> empty_ids;
  std::vector<float> empty_dis;
  tenann::ReserveTopK(&empty_ids, &empty_dis, 5);
  EXPECT_TRUE(empty_ids.empty());
  EXPECT_TRUE(empty_dis.empty());
}
