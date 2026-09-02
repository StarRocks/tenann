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
#include "tenann/index/index_str.h"
#include "tenann/index/parameter_serde.h"
#include "tenann/index/parameters.h"
#include "tenann/store/index_meta.h"
#include "tenann/store/index_type.h"

namespace tenann {

namespace {

IndexMeta MakeIvfSqMeta() {
  IndexMeta meta;
  meta.SetMetaVersion(0);
  meta.SetIndexFamily(IndexFamily::kVectorIndex);
  meta.SetIndexType(IndexType::kFaissIvfSq);
  meta.common_params()["dim"] = 32;
  meta.common_params()["metric_type"] = MetricType::kL2Distance;
  return meta;
}

}  // namespace

TEST(IvfSqParamsTest, Defaults) {
  auto meta = MakeIvfSqMeta();
  FaissIvfSqIndexParams index_params;
  FaissIvfSqSearchParams search_params;
  EXPECT_NO_THROW(FetchParameters(meta, &index_params));
  EXPECT_NO_THROW(FetchParameters(meta, &search_params));
  EXPECT_EQ(index_params.nlist, 16u);
  EXPECT_EQ(index_params.nbits, 8u);
  EXPECT_EQ(search_params.nprobe, 1u);
  EXPECT_EQ(search_params.max_codes, 0u);
}

TEST(IvfSqParamsTest, FetchValuesAndIndexString) {
  auto meta = MakeIvfSqMeta();
  meta.index_params()["nlist"] = 64;
  meta.index_params()["nbits"] = 4;
  meta.search_params()["nprobe"] = 8;
  meta.search_params()["max_codes"] = 1024;

  FaissIvfSqIndexParams index_params;
  FaissIvfSqSearchParams search_params;
  FetchParameters(meta, &index_params);
  FetchParameters(meta, &search_params);
  EXPECT_EQ(index_params.nlist, 64u);
  EXPECT_EQ(index_params.nbits, 4u);
  EXPECT_EQ(search_params.nprobe, 8u);
  EXPECT_EQ(search_params.max_codes, 1024u);
  EXPECT_EQ(IndexStr(meta), "ivf64sq4");
}

TEST(IvfSqParamsTest, RejectsUnsupportedNbits) {
  auto meta = MakeIvfSqMeta();
  meta.index_params()["nbits"] = 6;
  FaissIvfSqIndexParams params;
  EXPECT_THROW(FetchParameters(meta, &params), Error);
}

}  // namespace tenann
