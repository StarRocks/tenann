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

#include "test/test_base.h"

namespace tenann {

class FaissHnswIndexBuilderTest : public TestBase {
};

TEST_F(FaissHnswIndexBuilderTest, BuildWithPrimaryKeyImplWithArraySeqViewDataCol) {
  index_builder()->BuildWithPrimaryKey({id_view(), base_view()}, 0);

  std::shared_ptr<Index> index_ref = index_builder()->index_ref();
  EXPECT_TRUE(index_ref != nullptr);
  EXPECT_EQ(index_ref->index_type(), IndexType::kFaissHnsw);
}

TEST_F(FaissHnswIndexBuilderTest, BuildWithPrimaryKeyImplWithVlArraySeqViewDataCol) {
  index_builder()->BuildWithPrimaryKey({id_view(), base_vl_view()}, 0);

  std::shared_ptr<Index> index_ref = index_builder()->index_ref();
  EXPECT_TRUE(index_ref != nullptr);
  EXPECT_EQ(index_ref->index_type(), IndexType::kFaissHnsw);
}

TEST_F(FaissHnswIndexBuilderTest, BuildWithPrimaryKeyImpl_InvalidArgs) {
  // InputColumnsSizeZero
  {
    std::vector<SeqView> input_columns;
    EXPECT_THROW(
      index_builder()->BuildWithPrimaryKey(input_columns, 0),
      Error
    );
  }

  // InvalidDataColType
  EXPECT_THROW(
    // set SeqViewType::kPrimitiveSeqView to input_columns[data_col_index].seq_view_type
    index_builder()->BuildWithPrimaryKey({id_view(), id_view()}, 0),
    Error
  );

  // InvalidDataColElemType
  EXPECT_THROW(
    auto new_base_view = base_view();
    new_base_view.elem_type = PrimitiveType::kDoubleType;
    index_builder()->BuildWithPrimaryKey({id_view(), new_base_view}, 0),
    Error
  );

  // ContainsDimFalse
  EXPECT_THROW(
    auto new_meta = meta();
    new_meta.common_params().erase("dim");
    std::unique_ptr<IndexBuilder> faiss_hnsw_index_builder = std::make_unique<FaissHnswIndexBuilder>(new_meta);
    faiss_hnsw_index_builder->BuildWithPrimaryKey({id_view(), base_view()}, 0),
    Error
  );

  // ContainsMetricTypeFalse
  EXPECT_THROW(
    auto new_meta = meta();
    new_meta.common_params().erase("metric_type");
    std::unique_ptr<IndexBuilder> faiss_hnsw_index_builder = std::make_unique<FaissHnswIndexBuilder>(new_meta);
    faiss_hnsw_index_builder->BuildWithPrimaryKey({id_view(), base_view()}, 0),
    Error
  );

  // InvalidMetricType
  EXPECT_THROW(
    auto new_meta = meta();
    new_meta.common_params()["metric_type"] =  MetricType::kCosineSimilarity;
    std::unique_ptr<IndexBuilder> faiss_hnsw_index_builder = std::make_unique<FaissHnswIndexBuilder>(new_meta);
    faiss_hnsw_index_builder->BuildWithPrimaryKey({id_view(), base_view()}, 0),
    Error
  );

  // InvalidM, M == -1
  EXPECT_THROW(
    auto new_meta = meta();
    new_meta.index_params()["M"] = -1;
    //new_meta.index_params()["M"] = 100000000; run failed in DevCloud machine(32GB RAM)
    std::unique_ptr<IndexBuilder> faiss_hnsw_index_builder = std::make_unique<FaissHnswIndexBuilder>(new_meta);
    faiss_hnsw_index_builder->BuildWithPrimaryKey({id_view(), base_view()}, 0),
    Error
  );

  // DataColVlArraySeqViewSliceSizeError
  EXPECT_THROW(
    auto new_base_vl_view = base_vl_view();
    new_base_vl_view.offsets[1] = 0;
    index_builder()->BuildWithPrimaryKey({id_view(), new_base_vl_view}, 0),
    Error
  );
}

TEST_F(FaissHnswIndexBuilderTest, BuildImpl) {
  index_builder()->Build({base_view()});

  std::shared_ptr<Index> index_ref = index_builder()->index_ref();
  EXPECT_TRUE(index_ref != nullptr);
  EXPECT_EQ(index_ref->index_type(), IndexType::kFaissHnsw);
}

TEST_F(FaissHnswIndexBuilderTest, BuildImplWithVlArraySeqViewDataCol) {
  index_builder()->Build({base_vl_view()});

  std::shared_ptr<Index> index_ref = index_builder()->index_ref();
  EXPECT_TRUE(index_ref != nullptr);
  EXPECT_EQ(index_ref->index_type(), IndexType::kFaissHnsw);
}

TEST_F(FaissHnswIndexBuilderTest, BuildImpl_InvalidArgs) {
  // InputColumnsSizeZero
  {
    std::vector<SeqView> input_columns;
    EXPECT_THROW(
      index_builder()->Build(input_columns),
      Error
    );
  }

  // InvalidDataColType
  EXPECT_THROW(
    // set SeqViewType::kPrimitiveSeqView to input_columns[0].seq_view_type
    index_builder()->Build({id_view()}),
    Error
  );

  // InvalidDataColElemType
  EXPECT_THROW(
    auto new_base_view = base_view();
    new_base_view.elem_type = PrimitiveType::kDoubleType;
    index_builder()->Build({new_base_view}),
    Error
  );

  // ContainsDimFalse
  EXPECT_THROW(
    auto new_meta = meta();
    new_meta.common_params().erase("dim");
    std::unique_ptr<IndexBuilder> faiss_hnsw_index_builder = std::make_unique<FaissHnswIndexBuilder>(new_meta);
    faiss_hnsw_index_builder->Build({base_view()}),
    Error
  );

  // ContainsMetricTypeFalse
  EXPECT_THROW(
    auto new_meta = meta();
    new_meta.common_params().erase("metric_type");
    std::unique_ptr<IndexBuilder> faiss_hnsw_index_builder = std::make_unique<FaissHnswIndexBuilder>(new_meta);
    faiss_hnsw_index_builder->Build({base_view()}),
    Error
  );

  // InvalidMetricType
  EXPECT_THROW(
    auto new_meta = meta();
    new_meta.common_params()["metric_type"] =  MetricType::kCosineSimilarity;
    std::unique_ptr<IndexBuilder> faiss_hnsw_index_builder = std::make_unique<FaissHnswIndexBuilder>(new_meta);
    faiss_hnsw_index_builder->Build({base_view()}),
    Error
  );

  // InvalidM, M == -1
  EXPECT_THROW(
    auto new_meta = meta();
    new_meta.index_params()["M"] = -1;
    //new_meta.index_params()["M"] = 100000000; run failed in DevCloud machine(32GB RAM)
    std::unique_ptr<IndexBuilder> faiss_hnsw_index_builder = std::make_unique<FaissHnswIndexBuilder>(new_meta);
    faiss_hnsw_index_builder->Build({base_view()}),
    Error
  );

  // DataColVlArraySeqViewSliceSizeError
  EXPECT_THROW(
    auto new_base_vl_view = base_vl_view();
    new_base_vl_view.offsets[1] = 0;
    index_builder()->Build({new_base_vl_view}),
    Error
  );
}

TEST_F(FaissHnswIndexBuilderTest, FetchIdsTest) {
  int data_col_index;
  int64_t* fetched_ids;
  size_t num_ids;

  FaissHnswIndexBuilder builder(meta());

  // primary_key_column_index == 0
  builder.FetchIds({id_view(), base_view()}, 0, &data_col_index, &fetched_ids, &num_ids);
  EXPECT_EQ(data_col_index, 1);
  EXPECT_EQ(num_ids, id_view().size);
  for (size_t i = 0; i < num_ids; i++) {
    EXPECT_EQ(fetched_ids[i], reinterpret_cast<int64_t*>(id_view().data)[i]);
  }

  // primary_key_column_index == 1
  builder.FetchIds({base_view(), id_view()}, 1, &data_col_index, &fetched_ids, &num_ids);
  EXPECT_EQ(data_col_index, 0);
  EXPECT_EQ(num_ids, id_view().size);
  for (size_t i = 0; i < num_ids; i++) {
    EXPECT_EQ(fetched_ids[i], reinterpret_cast<int64_t*>(id_view().data)[i]);
  }

  // primary_key_column_index == 2, through T_DCHECK, stop at T_CHECK
  EXPECT_THROW(
    builder.FetchIds({id_view(), base_view()}, 2, &data_col_index, &fetched_ids, &num_ids),
    Error);

  // Invalid args: id_seq.seq_view_type == SeqViewType::kArraySeqView
  EXPECT_THROW(
    builder.FetchIds({base_view(), base_view()}, 0, &data_col_index, &fetched_ids, &num_ids),
    Error);

  // Invalid args: id_seq.seq_view.primitive_seq_view.elem_type == PrimitiveType::kFloatType
  EXPECT_THROW(
    auto new_id_view = id_view();
    new_id_view.elem_type = PrimitiveType::kFloatType;
    builder.FetchIds({new_id_view, base_view()}, 0, &data_col_index, &fetched_ids, &num_ids),
    Error);
}

}  // namespace tenann