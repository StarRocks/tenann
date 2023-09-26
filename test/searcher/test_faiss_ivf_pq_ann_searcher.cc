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

class FaissIvfPqAnnSearcherTest : public FaissTestBase {
 public:
  FaissIvfPqAnnSearcherTest() : FaissTestBase() {
    InitFaissIvfPqMeta();
    faiss_ivf_pq_index_builder_ = std::make_unique<FaissIvfPqIndexBuilder>(faiss_ivf_pq_meta_);
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

TEST_F(FaissIvfPqAnnSearcherTest, AnnSearch_CheckIsWork) {
  // Training and building take a lot of time.
  CreateAndWriteFaissIvfPqIndex();
  {
    // default search
    ReadIndexAndDefaultSearch();
    EXPECT_TRUE(IVFPQCheckResult());
  }

  {
    // nprobe = 1, recall rate < 0.8
    faiss_ivf_pq_meta().search_params()["nprobe"] = size_t(4 * sqrt(nb_));
    ann_searcher_->SetSearchParams(faiss_ivf_pq_meta().search_params());

    // search index
    result_ids_.clear();
    for (int i = 0; i < nq_; i++) {
      ann_searcher_->AnnSearch(query_view_[i], k_, result_ids_.data() + i * k_);
    }
    EXPECT_TRUE(IVFPQCheckResult());
  }

  {
    // nprobe = 4 * sqrt(nb_), recall rate > 0.8
    ann_searcher_->SetSearchParamItem(FAISS_SEARCHER_PARAMS_IVF_NPROBE, size_t(4 * sqrt(nb_)));
    ann_searcher_->SetSearchParamItem(FAISS_SEARCHER_PARAMS_IVF_MAX_CODES, size_t(0));
    ann_searcher_->SetSearchParamItem(FAISS_SEARCHER_PARAMS_IVF_PQ_SCAN_TABLE_THRESHOLD, size_t(0));
    ann_searcher_->SetSearchParamItem(FAISS_SEARCHER_PARAMS_IVF_PQ_POLYSEMOUS_HT, int(0));

    // search index
    result_ids_.clear();
    for (int i = 0; i < nq_; i++) {
      ann_searcher_->AnnSearch(query_view_[i], k_, result_ids_.data() + i * k_);
    }

    EXPECT_TRUE(IVFPQCheckResult());
  }
}

}  // namespace tenann