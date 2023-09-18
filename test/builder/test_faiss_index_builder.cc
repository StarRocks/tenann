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

class FaissIndexBuilderTest : public FaissTestBase {};

// for FaissIndexBuilder common test
class TmpFassIndexBuilder : public FaissIndexBuilder {
 public:
  using FaissIndexBuilder::FaissIndexBuilder;
  virtual ~TmpFassIndexBuilder() = default;

  T_FORBID_COPY_AND_ASSIGN(TmpFassIndexBuilder);
  T_FORBID_MOVE(TmpFassIndexBuilder);

 protected:
  IndexRef InitIndex() override { return nullptr; }
};

TEST_F(FaissIndexBuilderTest, ConstructorArgsError) {
  // missing dim
  EXPECT_THROW(auto new_meta = meta(); new_meta.common_params().erase("dim");
               auto faiss_index_builder = std::make_unique<TmpFassIndexBuilder>(new_meta), Error);

  // missing metric_type
  EXPECT_THROW(auto new_meta = meta(); new_meta.common_params().erase("metric_type");
               auto faiss_index_builder = std::make_unique<TmpFassIndexBuilder>(new_meta), Error);

  // metric_type is invalid
  EXPECT_THROW(auto new_meta = meta(); new_meta.common_params()["metric_type"] = MetricType::kCosineSimilarity;
               auto faiss_index_builder = std::make_unique<TmpFassIndexBuilder>(new_meta), Error);
}

TEST_F(FaissIndexBuilderTest, Add) {
  // not open
  EXPECT_THROW(std::make_unique<TmpFassIndexBuilder>(meta())->Add({base_view()}), Error);

  // add after close
  EXPECT_THROW(auto hnsw_index_builder = std::make_unique<TmpFassIndexBuilder>(meta());
               hnsw_index_builder->Open(); hnsw_index_builder->Close();
               hnsw_index_builder->Add({base_view()}), Error);

  // use_custom_row_id_(true), row_ids(nullptr)
  EXPECT_THROW(std::make_unique<TmpFassIndexBuilder>(meta())->EnableCustomRowId().Open().Add({base_view()}), Error);

  // use_custom_row_id_(false), row_ids(not null)
  EXPECT_THROW(std::make_unique<TmpFassIndexBuilder>(meta())->Open().Add({base_view()}, ids().data()), Error);

  // input_columns.size() != 1
  EXPECT_THROW(std::make_unique<TmpFassIndexBuilder>(meta())->Open().Add({}), Error);
  EXPECT_THROW(std::make_unique<TmpFassIndexBuilder>(meta())->Open().Add({base_view(), base_view()}), Error);

  // (input_seq_type == SeqViewType::kArraySeqView || input_seq_type == SeqViewType::kVlArraySeqView) == false
  EXPECT_THROW(std::make_unique<TmpFassIndexBuilder>(meta())->Open().Add({id_view()}), Error);

  // elem_type != PrimitiveType::kFloatType
  // ArraySeqView
  EXPECT_THROW(auto new_base_view = base_view(); new_base_view.elem_type = PrimitiveType::kDoubleType;
               std::make_unique<TmpFassIndexBuilder>(meta())->Open().Add({new_base_view}), Error);
  // VlArraySeqView
  EXPECT_THROW(auto new_base_vl_view = base_vl_view(); new_base_vl_view.elem_type = PrimitiveType::kDoubleType;
               std::make_unique<TmpFassIndexBuilder>(meta())->Open().Add({new_base_vl_view}), Error);
}

}  // namespace tenann