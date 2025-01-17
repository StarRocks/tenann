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

class FaissIvfPqIndexBuilderTest : public FaissTestBase {
 public:
  FaissIvfPqIndexBuilderTest() : FaissTestBase() {
    InitFaissIvfPqMeta();
    faiss_ivf_pq_index_builder_ = IndexFactory::CreateBuilderFromMeta(faiss_ivf_pq_meta_);
  }
};

TEST_F(FaissIvfPqIndexBuilderTest, Open) {
  {
    // open with pure memory
    auto ivf_pq_index_builder = std::make_unique<FaissIvfPqIndexBuilder>(faiss_ivf_pq_meta());
    auto index_ref = ivf_pq_index_builder->Open().index_ref();
    EXPECT_TRUE(index_ref != nullptr);
    EXPECT_EQ(index_ref->index_type(), IndexType::kFaissIvfPq);

    // reopen with pure memory
    EXPECT_THROW(ivf_pq_index_builder->Open(), Error);
  }

  {
    // open with path
    auto ivf_pq_index_builder = std::make_unique<FaissIvfPqIndexBuilder>(faiss_ivf_pq_meta());
    auto index_ref = ivf_pq_index_builder->Open(index_path()).index_ref();
    EXPECT_TRUE(index_ref != nullptr);
    EXPECT_EQ(index_ref->index_type(), IndexType::kFaissIvfPq);

    // reopen with pure memory
    EXPECT_THROW(ivf_pq_index_builder->Open(), Error);
  }
}

TEST_F(FaissIvfPqIndexBuilderTest, InitIndex) {
  // invalid M
  // new_meta.index_params()["M"] = 100000000; run failed in DevCloud machine(32GB RAM)
  EXPECT_THROW(auto new_meta = faiss_ivf_pq_meta(); new_meta.index_params()["M"] = -1;
               std::make_unique<FaissIvfPqIndexBuilder>(new_meta)->Open(), Error);
  // TODO: add "M", "efConstruction", "efSearch" limit UT
}

TEST_F(FaissIvfPqIndexBuilderTest, Add) {
  // TypedArraySeqView
  // use_custom_row_id_(true), null_map(not null)
  std::make_unique<FaissIvfPqIndexBuilder>(faiss_ivf_pq_meta())
      ->EnableCustomRowId()
      .Open()
      .Add({base_view()}, ids().data(), null_flags().data());
  // use_custom_row_id_(true), null_map(is null)
  std::make_unique<FaissIvfPqIndexBuilder>(faiss_ivf_pq_meta())
      ->EnableCustomRowId()
      .Open()
      .Add({base_view()}, ids().data());
  // inputs_live_longer_than_this(true)
  std::make_unique<FaissIvfPqIndexBuilder>(faiss_ivf_pq_meta())
      ->EnableCustomRowId()
      .Open()
      .Add({base_view()}, ids().data(), nullptr, true)
      .Flush()
      .Close();
  std::make_unique<FaissIvfPqIndexBuilder>(faiss_ivf_pq_meta())
      ->EnableCustomRowId()
      .Open()
      .Add({base_view()}, ids().data(), nullptr, true)
      .Add({base_view()}, ids().data(), nullptr, true)
      .Add({base_view()}, ids().data(), nullptr, true)
      .Flush()
      .Add({base_view()}, ids().data(), nullptr, true)
      .Close();
  // inputs_live_longer_than_this(false)
  std::make_unique<FaissIvfPqIndexBuilder>(faiss_ivf_pq_meta())
      ->EnableCustomRowId()
      .Open()
      .Add({base_view()}, ids().data(), nullptr, false)
      .Add({base_view()}, ids().data(), nullptr, false)
      .Add({base_view()}, ids().data(), nullptr, false)
      .Flush()
      .Add({base_view()}, ids().data(), nullptr, false)
      .Close();
  // use_custom_row_id_(false), null_map(not null)
  EXPECT_THROW(std::make_unique<FaissIvfPqIndexBuilder>(faiss_ivf_pq_meta())
                   ->Open()
                   .Add({base_view()}, nullptr, null_flags().data()),
               Error);
  // use_custom_row_id_(false), null_map(is null)
  std::make_unique<FaissIvfPqIndexBuilder>(faiss_ivf_pq_meta())->Open().Add({base_view()});

  // TypedVlArraySeqView
  // invalid dimension
  auto pre = base_vl_view().offsets[1];
  EXPECT_THROW(
      const_cast<uint32_t*>(base_vl_view().offsets)[1] = 0;
      std::make_unique<FaissIvfPqIndexBuilder>(faiss_ivf_pq_meta())->Open().Add({base_vl_view()}),
      Error);
  const_cast<uint32_t*>(base_vl_view().offsets)[1] = pre;
  // use_custom_row_id_(true), null_map(not null)
  std::make_unique<FaissIvfPqIndexBuilder>(faiss_ivf_pq_meta())
      ->EnableCustomRowId()
      .Open()
      .Add({base_vl_view()}, ids().data(), null_flags().data());
  // use_custom_row_id_(true), null_map(is null)
  std::make_unique<FaissIvfPqIndexBuilder>(faiss_ivf_pq_meta())
      ->EnableCustomRowId()
      .Open()
      .Add({base_vl_view()}, ids().data());
  // inputs_live_longer_than_this(true)
  std::make_unique<FaissIvfPqIndexBuilder>(faiss_ivf_pq_meta())
      ->EnableCustomRowId()
      .Open()
      .Add({base_vl_view()}, ids().data(), nullptr, true)
      .Flush()
      .Close();
  std::make_unique<FaissIvfPqIndexBuilder>(faiss_ivf_pq_meta())
      ->EnableCustomRowId()
      .Open()
      .Add({base_vl_view()}, ids().data(), nullptr, true)
      .Add({base_vl_view()}, ids().data(), nullptr, true)
      .Add({base_vl_view()}, ids().data(), nullptr, true)
      .Flush()
      .Add({base_vl_view()}, ids().data(), nullptr, true)
      .Close();
  // use_custom_row_id_(false), null_map(not null)
  EXPECT_THROW(std::make_unique<FaissIvfPqIndexBuilder>(faiss_ivf_pq_meta())
                   ->Open()
                   .Add({base_vl_view()}, nullptr, null_flags().data()),
               Error);
  // use_custom_row_id_(false), null_map(is null)
  std::make_unique<FaissIvfPqIndexBuilder>(faiss_ivf_pq_meta())->Open().Add({base_vl_view()});
}

}  // namespace tenann