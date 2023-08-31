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

#include <sys/time.h>

#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "tenann/common/error.h"
#include "tenann/builder/faiss_hnsw_index_builder.h"

namespace tenann {

class FaissHnswIndexBuilderTest : public ::testing::Test {
 public:
  std::unique_ptr<IndexBuilder>& index_builder() {
    return faiss_hnsw_index_builder_;
  }
  IndexMeta& meta() { return meta_; }
  PrimitiveSeqView& id_view() { return id_view_; }
  ArraySeqView& base_view() { return base_view_; }

 protected:
  void SetUp() override {
    meta_.SetMetaVersion(0);
    meta_.SetIndexFamily(IndexFamily::kVectorIndex);
    meta_.SetIndexType(IndexType::kFaissHnsw);
    meta_.common_params()["dim"] = 128;
    meta_.common_params()["is_vector_normed"] = false;
    meta_.common_params()["metric_type"] = MetricType::kL2Distance;
    meta_.index_params()["efConstruction"] = 40;
    meta_.index_params()["M"] = 32;
    meta_.search_params()["efSearch"] = 40;
    meta_.extra_params()["comments"] = "my comments";

    // dimension of the vectors to index
    uint32_t d = 128;
    // size of the database we plan to index
    size_t nb = 200;
    // size of the query vectors we plan to test
    size_t nq = 10;
    // index save path
    constexpr const char* index_path = "/tmp/faiss_hnsw_index";
    constexpr const char* index_with_primary_key_path = "/tmp/faiss_hnsw_index_with_ids";

    std::vector<int64_t> ids(nb);
    for (int i = 0; i < nb; i++) {
      ids[i] = i;
    }
    id_view_ = {.data = reinterpret_cast<uint8_t*>(ids.data()),
                                        .size = static_cast<uint32_t>(nb),
                                        .elem_type = PrimitiveType::kInt64Type};

    // generate data and query
    auto base = RandomVectors(nb, d);
    base_view_ = ArraySeqView{.data = reinterpret_cast<uint8_t*>(base.data()),
                                          .dim = d,
                                          .size = static_cast<uint32_t>(nb),
                                          .elem_type = PrimitiveType::kFloatType};

    auto query = RandomVectors(nq, d, /*seed=*/1);

    faiss_hnsw_index_builder_ = std::make_unique<FaissHnswIndexBuilder>(meta_);
  }

  void TearDown() override {}

  std::vector<float> RandomVectors(uint32_t n, uint32_t dim, int seed = 0) {
    std::mt19937 rng(seed);
    std::vector<float> data(n * dim);
    std::uniform_real_distribution<> distrib;
    for (size_t i = 0; i < n * dim; i++) {
      data[i] = distrib(rng);
    }
    return data;
  }

 private:
  IndexMeta meta_;
  PrimitiveSeqView id_view_;
  ArraySeqView base_view_;
  std::unique_ptr<IndexBuilder> faiss_hnsw_index_builder_;
};

TEST_F(FaissHnswIndexBuilderTest, BuildWithPrimaryKeyImpl) {
  index_builder()->BuildWithPrimaryKey({id_view(), base_view()}, 0);

  std::shared_ptr<Index> index_ref = index_builder()->index_ref();
  EXPECT_TRUE(index_ref != nullptr);
  EXPECT_EQ(index_ref->index_type(), IndexType::kFaissHnsw);
}

TEST_F(FaissHnswIndexBuilderTest, BuildImpl) {
  index_builder()->Build({base_view()});

  std::shared_ptr<Index> index_ref = index_builder()->index_ref();
  EXPECT_TRUE(index_ref != nullptr);
  EXPECT_EQ(index_ref->index_type(), IndexType::kFaissHnsw);
}

TEST_F(FaissHnswIndexBuilderTest, FetchIdsTest) {
  int data_col_index;
  int64_t* fetched_ids;
  size_t num_ids;

  FaissHnswIndexBuilder builder(meta());

  // test primary_key_column_index == 0
  builder.FetchIds({id_view(), base_view()}, 0, &data_col_index, &fetched_ids, &num_ids);
  EXPECT_EQ(data_col_index, 1);
  EXPECT_EQ(num_ids, id_view().size);
  for (size_t i = 0; i < num_ids; i++) {
    EXPECT_EQ(fetched_ids[i], reinterpret_cast<int64_t*>(id_view().data)[i]);
  }

  // test primary_key_column_index == 1
  builder.FetchIds({base_view(), id_view()}, 1, &data_col_index, &fetched_ids, &num_ids);
  EXPECT_EQ(data_col_index, 0);
  EXPECT_EQ(num_ids, id_view().size);
  for (size_t i = 0; i < num_ids; i++) {
    EXPECT_EQ(fetched_ids[i], reinterpret_cast<int64_t*>(id_view().data)[i]);
  }

  // test primary_key_column_index == 2
  EXPECT_THROW(
      builder.FetchIds({id_view(), base_view()}, 2, &data_col_index, &fetched_ids, &num_ids),
      Error);
}

}  // namespace tenann