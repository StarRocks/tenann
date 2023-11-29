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

class FaissIvfPqAnnSearcherTest : public FaissTestBase {
 public:
  FaissIvfPqAnnSearcherTest() : FaissTestBase() {
    InitFaissIvfPqMeta();
    faiss_ivf_pq_index_builder_ = IndexFactory::CreateBuilderFromMeta(faiss_ivf_pq_meta_);
  }
};

TEST_F(FaissIvfPqAnnSearcherTest, AnnSearch_InvalidArgs) {
  {
    IndexReaderRef index_reader = IndexFactory::CreateReaderFromMeta(faiss_ivf_pq_meta());
    auto ann_searcher = AnnSearcherFactory::CreateSearcherFromMeta(faiss_ivf_pq_meta());

    // index path not exist
    EXPECT_THROW(ann_searcher->SetIndexReader(index_reader)
                     .SetIndexCache(IndexCache::GetGlobalInstance())
                     .ReadIndex("not_exist_path", /*read_index_cache=*/false),
                 Error);

    // because ReadIndex fail, index_ref_ is null
    EXPECT_THROW(ann_searcher->AnnSearch(query_view()[0], k(), result_ids().data()), Error);
  }

  {
    // index_type() != IndexType::kFaissIvfPq
    EXPECT_THROW(
        IndexReaderRef index_reader = IndexFactory::CreateReaderFromMeta(faiss_ivf_pq_meta());
        auto ann_searcher = AnnSearcherFactory::CreateSearcherFromMeta(faiss_ivf_pq_meta());
        ann_searcher->SetIndexReader(index_reader)
            .SetIndexCache(IndexCache::GetGlobalInstance())
            .ReadIndex(index_with_primary_key_path(), /*read_index_cache=*/false);
        ann_searcher->index_ref()->SetIndexType(IndexType::kFaissHnsw);
        ann_searcher->AnnSearch(query_view()[0], k(), result_ids().data()), Error);
  }

  {
    CreateAndWriteFaissIvfPqIndex();
    // query_vector.elem_type != PrimitiveType::kFloatType
    auto double_type_query_view =
        PrimitiveSeqView{.data = reinterpret_cast<uint8_t*>(query_data().data()),
                         .size = d(),
                         .elem_type = PrimitiveType::kDoubleType};

    IndexReaderRef index_reader = IndexFactory::CreateReaderFromMeta(faiss_ivf_pq_meta());
    auto ann_searcher = AnnSearcherFactory::CreateSearcherFromMeta(faiss_ivf_pq_meta());
    ann_searcher->SetIndexReader(index_reader)
        .SetIndexCache(IndexCache::GetGlobalInstance())
        .ReadIndex(index_with_primary_key_path(), /*read_index_cache=*/false);
    EXPECT_THROW(ann_searcher->AnnSearch(double_type_query_view, k(), result_ids().data()), Error);
  }
}

TEST_F(FaissIvfPqAnnSearcherTest, AnnSearch_Check_IndexIvfPq_IsWork) {
  // Training and building take a lot of time.
  auto start = std::chrono::high_resolution_clock::now();
  CreateAndWriteFaissIvfPqIndex(true);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  printf("IVFPQ创建索引执行时间:  %d 毫秒\n", duration.count());

  {
    // default search
    ReadIndexAndDefaultSearch();
    EXPECT_TRUE(RecallCheckResult_80Percent());
  }

  {
    // nprobe = 1, recall rate < 0.8
    faiss_ivf_pq_meta().search_params()["nprobe"] = size_t(4 * sqrt(nb_));
    faiss_ivf_pq_meta().search_params()["max_codes"] = 0;
    faiss_ivf_pq_meta().search_params()["scan_table_threshold"] = 0;
    faiss_ivf_pq_meta().search_params()["polysemous_ht"] = 0;
    ann_searcher_->SetSearchParams(faiss_ivf_pq_meta().search_params());

    // search index
    result_ids_.clear();
    for (int i = 0; i < nq_; i++) {
      ann_searcher_->AnnSearch(query_view_[i], k_, result_ids_.data() + i * k_);
    }
    EXPECT_TRUE(RecallCheckResult_80Percent());
  }

  {
    // nprobe = 4 * sqrt(nb_), recall rate > 0.8
    ann_searcher_->SetSearchParamItem(FaissIvfPqSearchParams::nprobe_key, size_t(4 * sqrt(nb_)));
    ann_searcher_->SetSearchParamItem(FaissIvfPqSearchParams::max_codes_key, size_t(0));
    ann_searcher_->SetSearchParamItem(FaissIvfPqSearchParams::scan_table_threshold_key, size_t(0));
    ann_searcher_->SetSearchParamItem(FaissIvfPqSearchParams::polysemous_ht_key, int(0));

    // search index
    result_ids_.clear();
    for (int i = 0; i < nq_; i++) {
      ann_searcher_->AnnSearch(query_view_[i], k_, result_ids_.data() + i * k_);
    }

    EXPECT_TRUE(RecallCheckResult_80Percent());
  }
}

TEST_F(FaissIvfPqAnnSearcherTest, AnnSearch_Check_ID_Filter_IsWork) {
  // Training and building take a lot of time.
  auto start = std::chrono::high_resolution_clock::now();
  CreateAndWriteFaissIvfPqIndex(true, id_filter_count_);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  printf("IVFPQ创建索引执行时间:  %d 毫秒\n", duration.count());
  ReadIndexAndDefaultSearch();

  {
    // IdFilter 判定全为不感兴趣的，返回值应全为 -1
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
    // ArrayIdFilter 只对前 id_filter_count_ 个 ids 感兴趣，应完全匹配
    RangeIdFilter id_filter(0, id_filter_count_, false);
    result_ids_.clear();
    for (int i = 0; i < nq_; i++) {
      ann_searcher_->AnnSearch(query_view_[i], k_, result_ids_.data() + i * k_, &id_filter);
    }
    EXPECT_TRUE(RecallCheckResult_80Percent());
  }

  // test ArrayIdFilter
  {
    // ArrayIdFilter 只对前 id_filter_count_ 个 ids 感兴趣，应完全匹配
    ArrayIdFilter id_filter(ids_.data(), id_filter_count_);
    result_ids_.clear();
    for (int i = 0; i < nq_; i++) {
      ann_searcher_->AnnSearch(query_view_[i], k_, result_ids_.data() + i * k_, &id_filter);
    }
    EXPECT_TRUE(RecallCheckResult_80Percent());
  }

  // test BatchIdFilter
  {
    // BatchIdFilter 只对前 id_filter_count_ 个 ids 感兴趣，应完全匹配
    BatchIdFilter id_filter(ids_.data(), id_filter_count_);
    result_ids_.clear();
    for (int i = 0; i < nq_; i++) {
      ann_searcher_->AnnSearch(query_view_[i], k_, result_ids_.data() + i * k_, &id_filter);
    }
    EXPECT_TRUE(RecallCheckResult_80Percent());
  }

  // test BitmapIdFilter
  {
    // BitmapIdFilter 只对前 id_filter_count_ 个 ids 感兴趣
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

}  // namespace tenann