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

#include "tenann/index/parameters.h"
#include "test/faiss_test_base.h"

namespace tenann {

class FaissHnswAnnSearcherTest : public FaissTestBase {
 public:
  FaissHnswAnnSearcherTest() : FaissTestBase() {
    InitFaissHnswMeta();
    faiss_hnsw_index_builder_ = IndexFactory::CreateBuilderFromMeta(faiss_hnsw_meta_);
  }
};

TEST_F(FaissHnswAnnSearcherTest, AnnSearch_InvalidArgs) {
  CreateAndWriteFaissHnswIndex(true);

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
  {
    CreateAndWriteFaissHnswIndex(true);
    ReadIndexAndDefaultSearch();
    EXPECT_TRUE(RecallCheckResult_80Percent());
  }

  {
    // efSearch = 1, recall rate < 0.8
    faiss_hnsw_meta().search_params()["efSearch"] = int(1);
    ann_searcher_->SetSearchParams(faiss_hnsw_meta().search_params());

    // search index
    result_ids_.clear();
    for (int i = 0; i < nq_; i++) {
      ann_searcher_->AnnSearch(query_view_[i], k_, result_ids_.data() + i * k_);
    }
    EXPECT_FALSE(RecallCheckResult_80Percent());
  }

  {
    // efSearch = 40, recall rate > 0.8
    ann_searcher_->SetSearchParamItem(FaissHnswSearchParams::efSearch_key, int(40));
    ann_searcher_->SetSearchParamItem(FaissHnswSearchParams::check_relative_distance_key, true);

    // search index
    result_ids_.clear();
    for (int i = 0; i < nq_; i++) {
      ann_searcher_->AnnSearch(query_view_[i], k_, result_ids_.data() + i * k_);
    }
    EXPECT_TRUE(RecallCheckResult_80Percent());
  }
}

TEST_F(FaissHnswAnnSearcherTest, AnnSearch_Check_ID_Filter_IsWork) {
  {
    CreateAndWriteFaissHnswIndex(true, id_filter_count_);
    ReadIndexAndDefaultSearch();
  }

  {
    // 构造 IdFilter 对所有 ids 不感兴趣，搜索返回值应全为 -1
    class DerivedIdFilter : public IdFilter {
     public:
      bool IsMember(idx_t id) const override { return false; }
      ~DerivedIdFilter() override = default;
    } id_filter;  // 实例化匿名类的对象
    // search index
    result_ids_.clear();
    for (int i = 0; i < nq_; i++) {
      ann_searcher_->AnnSearch(query_view_[i], k_, result_ids_.data() + i * k_, &id_filter);
    }
    EXPECT_TRUE(std::all_of(result_ids_.data(), result_ids_.data() + nq_ * k_,
                            [](int64_t element) { return element == -1; }));
  }

  // test RangeIdFilter
  {
    // 构造 RangeIdFilter 只对[0, id_filter_count_)范围的 ids 感兴趣
    RangeIdFilter id_filter(0, id_filter_count_, false);
    result_ids_.clear();
    for (int i = 0; i < nq_; i++) {
      ann_searcher_->AnnSearch(query_view_[i], k_, result_ids_.data() + i * k_, &id_filter);
    }
    EXPECT_TRUE(RecallCheckResult_80Percent());
  }

  // test ArrayIdFilter
  {
    // 构造 ArrayIdFilter 只对[0, id_filter_count_)范围的 ids 感兴趣
    ArrayIdFilter id_filter(ids_.data(), id_filter_count_);
    result_ids_.clear();
    for (int i = 0; i < nq_; i++) {
      ann_searcher_->AnnSearch(query_view_[i], k_, result_ids_.data() + i * k_, &id_filter);
    }
    EXPECT_TRUE(RecallCheckResult_80Percent());
  }

  // test BatchIdFilter
  {
    // 构造 BatchIdFilter 只对[0, id_filter_count_)范围的 ids 感兴趣
    BatchIdFilter id_filter(ids_.data(), id_filter_count_);
    result_ids_.clear();
    for (int i = 0; i < nq_; i++) {
      ann_searcher_->AnnSearch(query_view_[i], k_, result_ids_.data() + i * k_, &id_filter);
    }
    EXPECT_TRUE(RecallCheckResult_80Percent());
  }

  // test BitmapIdFilter
  {
    // 构造 BitmapIdFilter 只对[0, id_filter_count_)范围的 ids 感兴趣
    std::vector<uint8_t> bitmap((nb_ + 7) / 8, 0);
    for (int i = 0; i < id_filter_count_ && i < nb_; ++i) {
      uint64_t id = ids_[i];
      bitmap[id >> 3] |= (1 << (id & 7));
    }
    BitmapIdFilter id_filter(bitmap.data(), bitmap.size());
    result_ids_.clear();
    for (int i = 0; i < nq_; i++) {
      ann_searcher_->AnnSearch(query_view_[i], k_, result_ids_.data() + i * k_, &id_filter);
    }
    EXPECT_TRUE(RecallCheckResult_80Percent());
  }
}

TEST_F(FaissHnswAnnSearcherTest, AnnSearch_Check_IndexHNSW_IsWork) {
  CreateAndWriteFaissHnswIndex(false);

  {
    ReadIndexAndDefaultSearch();
    EXPECT_TRUE(RecallCheckResult_80Percent());
  }
}

}  // namespace tenann