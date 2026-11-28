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

#include "test/faiss_test_base.h"

namespace tenann {

class FaissHnswIndexBuilderTest : public FaissTestBase {
 public:
  FaissHnswIndexBuilderTest() : FaissTestBase() {
    InitFaissHnswMeta();
    faiss_hnsw_index_builder_ = IndexFactory::CreateBuilderFromMeta(faiss_hnsw_meta_);
  }
};

TEST_F(FaissHnswIndexBuilderTest, Open) {
  {
    // open with pure memory
    auto hnsw_index_builder = std::make_unique<FaissHnswIndexBuilder>(faiss_hnsw_meta());
    auto index_ref = hnsw_index_builder->Open().index_ref();
    EXPECT_TRUE(index_ref != nullptr);
    EXPECT_EQ(index_ref->index_type(), IndexType::kFaissHnsw);

    // reopen with pure memory
    EXPECT_THROW(hnsw_index_builder->Open(), Error);
  }

  {
    // open with path
    auto hnsw_index_builder = std::make_unique<FaissHnswIndexBuilder>(faiss_hnsw_meta());
    auto index_ref = hnsw_index_builder->Open(index_path()).index_ref();
    EXPECT_TRUE(index_ref != nullptr);
    EXPECT_EQ(index_ref->index_type(), IndexType::kFaissHnsw);

    // reopen with pure memory
    EXPECT_THROW(hnsw_index_builder->Open(), Error);
  }
}

TEST_F(FaissHnswIndexBuilderTest, InitIndex) {
  // invalid M
  // new_meta.index_params()["M"] = 100000000; run failed in DevCloud machine(32GB RAM)
  EXPECT_THROW(auto new_meta = faiss_hnsw_meta(); new_meta.index_params()["M"] = -1;
               std::make_unique<FaissHnswIndexBuilder>(new_meta)->Open(), Error);

  // TODO: add "M", "efConstruction", "efSearch" limit UT
}

TEST_F(FaissHnswIndexBuilderTest, Add) {
  // TypedArraySeqView
  // use_custom_row_id_(true), null_map(not null)
  std::make_unique<FaissHnswIndexBuilder>(faiss_hnsw_meta())
      ->EnableCustomRowId()
      .Open()
      .Add({base_view()}, ids().data(), null_flags().data());
  // use_custom_row_id_(true), null_map(is null)
  std::make_unique<FaissHnswIndexBuilder>(faiss_hnsw_meta())
      ->EnableCustomRowId()
      .Open()
      .Add({base_view()}, ids().data());
  // use_custom_row_id_(false), null_map(not null)
  EXPECT_THROW(std::make_unique<FaissHnswIndexBuilder>(faiss_hnsw_meta())
                   ->Open()
                   .Add({base_view()}, nullptr, null_flags().data()),
               Error);
  // use_custom_row_id_(false), null_map(is null)

  std::make_unique<FaissHnswIndexBuilder>(faiss_hnsw_meta())->Open().Add({base_view()});

  // TypedVlArraySeqView
  // invalid dimension
  auto pre = base_vl_view().offsets[1];
  EXPECT_THROW(
      const_cast<uint32_t*>(base_vl_view().offsets)[1] = 0;
      std::make_unique<FaissHnswIndexBuilder>(faiss_hnsw_meta())->Open().Add({base_vl_view()}),
      Error);
  const_cast<uint32_t*>(base_vl_view().offsets)[1] = pre;
  // use_custom_row_id_(true), null_map(not null)
  std::make_unique<FaissHnswIndexBuilder>(faiss_hnsw_meta())
      ->EnableCustomRowId()
      .Open()
      .Add({base_vl_view()}, ids().data(), null_flags().data());
  // use_custom_row_id_(true), null_map(is null)
  std::make_unique<FaissHnswIndexBuilder>(faiss_hnsw_meta())
      ->EnableCustomRowId()
      .Open()
      .Add({base_vl_view()}, ids().data());
  // use_custom_row_id_(false), null_map(not null)
  EXPECT_THROW(std::make_unique<FaissHnswIndexBuilder>(faiss_hnsw_meta())
                   ->Open()
                   .Add({base_vl_view()}, nullptr, null_flags().data()),
               Error);
  // use_custom_row_id_(false), null_map(is null)
  std::make_unique<FaissHnswIndexBuilder>(faiss_hnsw_meta())->Open().Add({base_vl_view()});
}

TEST_F(FaissHnswIndexBuilderTest, AddRejectsNonFloatVlArraySeqView) {
  // Verify that FaissIndexBuilder::Add inspects only the active union member
  // when validating element types. Here size == 10 aliases PrimitiveType::kFloatType
  // at ArraySeqView::elem_type, which previously masked the invalid VlArraySeqView
  // element type and triggered union UB under sanitiser builds.
  constexpr uint32_t kNumRows = 10;
  std::vector<int32_t> int_data(kNumRows * d(), 0);
  VlArraySeqView invalid_vl_view{
      .data = reinterpret_cast<const uint8_t*>(int_data.data()),
      .offsets = offsets().data(),
      .size = kNumRows,
      .elem_type = PrimitiveType::kInt32Type,
  };

  auto builder = std::make_unique<FaissHnswIndexBuilder>(faiss_hnsw_meta());
  builder->Open();
  EXPECT_ANY_THROW(builder->Add({invalid_vl_view}));

  // Also verify non-float ArraySeqView is rejected without reading inactive union fields.
  ArraySeqView invalid_array_view{
      .data = reinterpret_cast<const uint8_t*>(int_data.data()),
      .dim = d(),
      .size = kNumRows,
      .elem_type = PrimitiveType::kInt32Type,
  };
  auto array_builder = std::make_unique<FaissHnswIndexBuilder>(faiss_hnsw_meta());
  array_builder->Open();
  EXPECT_ANY_THROW(array_builder->Add({invalid_array_view}));

  // Verify a valid VlArraySeqView (with size > 15) succeeds cleanly
  // without reading the inactive ArraySeqView union member.
  constexpr uint32_t kValidNumRows = 20;
  VlArraySeqView valid_vl_view{
      .data = reinterpret_cast<const uint8_t*>(base().data()),
      .offsets = offsets().data(),
      .size = kValidNumRows,
      .elem_type = PrimitiveType::kFloatType,
  };
  auto valid_builder = std::make_unique<FaissHnswIndexBuilder>(faiss_hnsw_meta());
  valid_builder->Open().Add({valid_vl_view});
}

}  // namespace tenann