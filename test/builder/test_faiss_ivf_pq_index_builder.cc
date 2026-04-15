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

#include "faiss/Index.h"
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

// =============== SetFlushThresholdRows / partial flush ===============
//
// Threshold below uses small row counts so the test triggers partial flush
// without needing a large dataset.

namespace {
// Read ntotal off the underlying faiss index (works for any Index subclass).
int64_t FaissNtotal(const std::shared_ptr<Index>& ref) {
  return static_cast<faiss::Index*>(ref->index_raw())->ntotal;
}
}  // namespace

// SetFlushThresholdRows is a chainable setter on the IndexBuilder base class.
TEST_F(FaissIvfPqIndexBuilderTest, SetFlushThresholdRowsIsChainable) {
  auto builder = std::make_unique<FaissIvfPqIndexBuilder>(faiss_ivf_pq_meta());
  EXPECT_EQ(&builder->SetFlushThresholdRows(100'000), builder.get());
  EXPECT_EQ(&builder->SetFlushThresholdRows(0), builder.get());
}

// MaybeFlushBuffer must be a no-op while the index is still untrained: the
// buffer accumulates the full sample, Flush() trains on it, and only then
// drains. Setting an aggressive 1-row threshold must NOT change the
// untrained behaviour.
TEST_F(FaissIvfPqIndexBuilderTest, FlushThresholdNoOpForUntrained) {
  auto builder = std::make_unique<FaissIvfPqIndexBuilder>(faiss_ivf_pq_meta());
  builder->SetFlushThresholdRows(1)
      .EnableCustomRowId()
      .Open()
      .Add({base_view()}, ids().data(), null_flags().data());
  // Nothing has been trained or added yet — partial flush did not fire.
  EXPECT_EQ(FaissNtotal(builder->index_ref()), 0);
  builder->Flush();
  // After Flush the index is trained and contains the non-null rows.
  EXPECT_GT(FaissNtotal(builder->index_ref()), 0);
  builder->Close();
}

// Once the index is trained (via the first Flush), subsequent Add() calls
// should hit MaybeFlushBuffer and drain on the spot when the buffer crosses
// the threshold. With a 1-row threshold the very first non-null row drains.
TEST_F(FaissIvfPqIndexBuilderTest, PartialFlushDrainsTrainedIndex) {
  auto builder = std::make_unique<FaissIvfPqIndexBuilder>(faiss_ivf_pq_meta());
  builder->EnableCustomRowId()
      .Open()
      .Add({base_view()}, ids().data(), null_flags().data())
      .Flush();
  int64_t ntotal_after_train = FaissNtotal(builder->index_ref());
  EXPECT_GT(ntotal_after_train, 0);

  builder->SetFlushThresholdRows(1);
  builder->Add({base_view()}, ids().data(), null_flags().data());
  // MaybeFlushBuffer should have drained mid-Add — ntotal grew without a
  // separate Flush() call.
  int64_t ntotal_after_add = FaissNtotal(builder->index_ref());
  EXPECT_GT(ntotal_after_add, ntotal_after_train);

  // Final Flush is a no-op because the buffer is already drained.
  builder->Flush();
  EXPECT_EQ(FaissNtotal(builder->index_ref()), ntotal_after_add);
  builder->Close();
}

// SetFlushThresholdRows(0) disables intermediate flushing even after the
// index is trained — the buffer accumulates until the next explicit Flush().
TEST_F(FaissIvfPqIndexBuilderTest, FlushThresholdZeroDisablesPartialFlush) {
  auto builder = std::make_unique<FaissIvfPqIndexBuilder>(faiss_ivf_pq_meta());
  builder->EnableCustomRowId()
      .Open()
      .Add({base_view()}, ids().data(), null_flags().data())
      .Flush();
  int64_t ntotal_after_train = FaissNtotal(builder->index_ref());

  builder->SetFlushThresholdRows(0);
  builder->Add({base_view()}, ids().data(), null_flags().data());
  // No partial flush — ntotal unchanged after Add.
  EXPECT_EQ(FaissNtotal(builder->index_ref()), ntotal_after_train);

  builder->Flush();
  // The accumulated buffer is drained by the explicit Flush.
  EXPECT_GT(FaissNtotal(builder->index_ref()), ntotal_after_train);
  builder->Close();
}

// Flush now clears the row buffer at the end so a second call is a safe
// no-op rather than re-adding the same rows.
TEST_F(FaissIvfPqIndexBuilderTest, FlushIsIdempotent) {
  auto builder = std::make_unique<FaissIvfPqIndexBuilder>(faiss_ivf_pq_meta());
  builder->EnableCustomRowId()
      .Open()
      .Add({base_view()}, ids().data(), null_flags().data())
      .Flush();
  int64_t ntotal_after_first = FaissNtotal(builder->index_ref());

  EXPECT_NO_THROW(builder->Flush());
  EXPECT_EQ(FaissNtotal(builder->index_ref()), ntotal_after_first);
  builder->Close();
}

}  // namespace tenann