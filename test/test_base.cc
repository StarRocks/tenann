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

#include "test/test_base.h"

namespace tenann {

void TestBase::SetUp() {
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

  ids_.resize(nb_);
  for (int i = 0; i < nb_; i++) {
    ids_[i] = i;
  }
  id_view_ = PrimitiveSeqView{.data = reinterpret_cast<uint8_t*>(ids_.data()),
                              .size = static_cast<uint32_t>(nb_),
                              .elem_type = PrimitiveType::kInt64Type};

  // generate data and query
  base_ = RandomVectors(nb_, d_);
  base_view_ = ArraySeqView{.data = reinterpret_cast<uint8_t*>(base_.data()),
                            .dim = d_,
                            .size = static_cast<uint32_t>(nb_),
                            .elem_type = PrimitiveType::kFloatType};

  int offset_num = nb_ + 1;
  offsets_.resize(offset_num);
  offsets_[0] = 0;
  for (int i = 1; i < offset_num; i++) {
    offsets_[i] = offsets_[i - 1] + d_;
  }
  base_vl_view_ = VlArraySeqView{.data = reinterpret_cast<uint8_t*>(base_.data()),
                                 .offsets = offsets_.data(),
                                 .size = static_cast<uint32_t>(nb_),
                                 .elem_type = PrimitiveType::kFloatType};

  query_ = RandomVectors(nq_, d_, /*seed=*/1);

  faiss_hnsw_index_builder_ = std::make_unique<FaissHnswIndexBuilder>(meta_);

  result_ids_.resize(nq_ * k_);
  accurate_query_result_ids_.resize(nq_ * k_);

  InitAccurateQueryResult();
}

std::vector<float> TestBase::RandomVectors(uint32_t n, uint32_t dim, int seed) {
  std::mt19937 rng(seed);
  std::vector<float> data(n * dim);
  std::uniform_real_distribution<> distrib;
  for (size_t i = 0; i < n * dim; i++) {
    data[i] = distrib(rng);
  }
  return data;
}

float TestBase::EuclideanDistance(const float* v1, const float* v2) {
  float sum = 0.0;
  for (size_t i = 0; i < d_; i++) {
      float diff = *(v2 + i) - *(v1 + i);
      sum += diff * diff;
  }

  return std::sqrt(sum);
}

void TestBase::InitAccurateQueryResult() {
  // search index
  for (int i = 0; i < nq_; i++) {
    std::vector<std::pair<int, double>> distances;
    for ( int j = 0; j< nb_; j++) {
      float distance = EuclideanDistance(query_.data() + i * d_, base_.data() + j * d_);
      distances.push_back(std::make_pair(j, distance));
    }

    std::sort(distances.begin(), distances.end(), [](const auto& a, const auto& b) {
      return a.second < b.second;
    });
    distances.resize(k_);
    
    for ( int j = 0; j < k_; j++) {
      accurate_query_result_ids_[i * k_ + j] = distances[j].first;
    }
  }
}

void TestBase::CreateAndWriteIndex() {
  index_writer_ = IndexFactory::CreateWriterFromMeta(meta_);

  faiss_hnsw_index_builder_->SetIndexWriter(index_writer_)
      .SetIndexCache(IndexCache::GetGlobalInstance())
      .BuildWithPrimaryKey({id_view_, base_view_}, 0)
      .WriteIndex(index_with_primary_key_path_, /*write_index_cache=*/true);
}

void TestBase::ReadIndexAndDefaultSearch() {
  index_reader_ = IndexFactory::CreateReaderFromMeta(meta_);
  ann_searcher_ = AnnSearcherFactory::CreateSearcherFromMeta(meta_);

  // load index from disk file
  ann_searcher_->SetIndexReader(index_reader_)
      .SetIndexCache(IndexCache::GetGlobalInstance())
      .ReadIndex(index_with_primary_key_path_, /*read_index_cache=*/true);

  result_ids_.resize(nq_ * k_);

  // search index
  for (int i = 0; i < nq_; i++) {
    auto query_view =
        PrimitiveSeqView{.data = reinterpret_cast<uint8_t*>(query_.data() + i * d_),
                         .size = d_,
                         .elem_type = PrimitiveType::kFloatType};
    ann_searcher_->AnnSearch(query_view, k_, result_ids_.data() + i * k_);
  }
}

// log output: build_Release/Testing/Temporary/LastTest.log
bool TestBase::CheckResult() {
  for (int i = 0; i < nq_; i++) {
    printf("query %d:\n", i);
    for (int j = 0; j < k_; j++) {
      printf("%d vs %d\n", result_ids_[i * k_ + j], accurate_query_result_ids_[i * k_ + j]);
      if (result_ids_[i * k_ + j] != accurate_query_result_ids_[i * k_ + j]) {
        return false;
      }
    }
    printf("\n");
  }
  return true;
}

}  // namespace tenann