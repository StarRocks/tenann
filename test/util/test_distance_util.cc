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