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

class FaissHnswAnnSearcherTest : public FaissTestBase {
 public:
    FaissHnswAnnSearcherTest() : FaissTestBase() {
      InitFaissHnswMeta();
      faiss_hnsw_index_builder_ = std::make_unique<FaissHnswIndexBuilder>(faiss_hnsw_meta_);
    }
};

TEST_F(FaissHnswAnnSearcherTest, AnnSearch_InvalidArgs) {
  CreateAndWriteFaissHnswIndex();

  {
    IndexReaderRef index_reader = IndexFactory::CreateReaderFromMeta(faiss_hnsw_meta());
    auto ann_searcher = AnnSearcherFactory::CreateSearcherFromMeta(faiss_hnsw_meta());

    // index path not exist
    EXPECT_THROW(ann_searcher->SetIndexReader(index_reader)
                     .SetIndexCache(IndexCache::GetGlobalInstance())
                     .ReadIndex("not_exist_path", /*read_index_cache=*/false),
                 Error);

    // because ReadIndex fail, index_ref_ is null
    EXPECT_THROW(ann_searcher->AnnSearch(query_view()[0], k(), result_ids().data()), Error);
  }

  {
    // index_type() != IndexType::kFaissHnsw
    EXPECT_THROW(
        IndexReaderRef index_reader = IndexFactory::CreateReaderFromMeta(faiss_hnsw_meta());
        auto ann_searcher = AnnSearcherFactory::CreateSearcherFromMeta(faiss_hnsw_meta());
        ann_searcher->SetIndexReader(index_reader)
            .SetIndexCache(IndexCache::GetGlobalInstance())
            .ReadIndex(index_with_primary_key_path(), /*read_index_cache=*/false);
        ann_searcher->index_ref()->SetIndexType(IndexType::kFaissIvfPq);
        ann_searcher->AnnSearch(query_view()[0], k(), result_ids().data()), Error);
  }

  {
    // query_vector.elem_type != PrimitiveType::kFloatType
    auto double_type_query_view =
        PrimitiveSeqView{.data = reinterpret_cast<uint8_t*>(query_data().data()),
                         .size = d(),
                         .elem_type = PrimitiveType::kDoubleType};

    IndexReaderRef index_reader = IndexFactory::CreateReaderFromMeta(faiss_hnsw_meta());
    auto ann_searcher = AnnSearcherFactory::CreateSearcherFromMeta(faiss_hnsw_meta());
    ann_searcher->SetIndexReader(index_reader)
        .SetIndexCache(IndexCache::GetGlobalInstance())
        .ReadIndex(index_with_primary_key_path(), /*read_index_cache=*/false);
    EXPECT_THROW(ann_searcher->AnnSearch(double_type_query_view, k(), result_ids().data()), Error);
  }
}

TEST_F(FaissHnswAnnSearcherTest, AnnSearch_Check_IDMap_HNSW_IsWork) {
  CreateAndWriteFaissHnswIndex(true);

  {
    // default search index, efSearch = 16, recall rate > 0.8
    ReadIndexAndDefaultSearch();
    // TODO: fix this test
    // EXPECT_TRUE(RecallCheckResult_80Percent());
  }

  {
    // efSearch = 1, recall rate < 0.8
    faiss_hnsw_meta().search_params()["efSearch"] = int(1);
    ann_searcher_->SetDefaultSearchParams(faiss_hnsw_meta().search_params());

    // search index
    result_ids_.clear();
    for (int i = 0; i < nq_; i++) {
      ann_searcher_->AnnSearch(query_view_[i], k_, result_ids_.data() + i * k_);
    }
    EXPECT_FALSE(RecallCheckResult_80Percent());
  }

  {
    // TODO: hnsw 暂未支持传入 searchParams, 预备 ut
    // efSearch = 40, recall rate > 0.8
    ann_searcher_->SetDefaultSearchParamItem(FAISS_SEARCHER_PARAMS_HNSW_EF_SEARCH, int(40));
    ann_searcher_->SetDefaultSearchParamItem(FAISS_SEARCHER_PARAMS_HNSW_CHECK_RELATIVE_DISTANCE,
                                             true);

    // search index
    result_ids_.clear();
    for (int i = 0; i < nq_; i++) {
      ann_searcher_->AnnSearch(query_view_[i], k_, result_ids_.data() + i * k_);
    }
    // TODO: fix this test
    // EXPECT_TRUE(RecallCheckResult_80Percent());
  }
}

// TODO: 不使用 IDMap 的情况下召回率为 0，需要排查原因
TEST_F(FaissHnswAnnSearcherTest, AnnSearch_Check_IndexHNSW_IsWork) {
  CreateAndWriteFaissHnswIndex(false);

  {
    ReadIndexAndDefaultSearch();
    EXPECT_FALSE(RecallCheckResult_80Percent());
  }
}

}  // namespace tenann