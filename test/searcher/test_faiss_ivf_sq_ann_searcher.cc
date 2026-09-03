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

#include <algorithm>
#include <random>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "tenann/common/seq_view.h"
#include "tenann/factory/ann_searcher_factory.h"
#include "tenann/factory/index_factory.h"
#include "tenann/searcher/faiss_ivf_sq_ann_searcher.h"
#include "tenann/searcher/id_filter.h"
#include "tenann/store/index_meta.h"

namespace tenann {

namespace {

constexpr uint32_t kDim = 16;
constexpr uint32_t kNumRows = 1000;
constexpr int64_t kIdOffset = 10000;

IndexMeta MakeIvfSqMeta(MetricType metric_type, const std::string& cosine_backend = "l2") {
  IndexMeta meta;
  meta.SetMetaVersion(0);
  meta.SetIndexFamily(IndexFamily::kVectorIndex);
  meta.SetIndexType(IndexType::kFaissIvfSq);
  meta.common_params()["dim"] = static_cast<int>(kDim);
  meta.common_params()["metric_type"] = metric_type;
  meta.common_params()["is_vector_normed"] = false;
  meta.index_params()["nlist"] = 16;
  meta.index_params()["nbits"] = 8;
  meta.search_params()["nprobe"] = 16;
  meta.search_params()["max_codes"] = 0;
  meta.index_writer_options()["write_index_cache"] = false;
  meta.index_writer_options()["cosine_backend"] = cosine_backend;
  meta.index_reader_options()["cache_index_file"] = false;
  return meta;
}

std::vector<float> RandomVectors() {
  std::mt19937 rng(11);
  std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
  std::vector<float> vectors(static_cast<size_t>(kNumRows) * kDim);
  for (auto& value : vectors) {
    value = distribution(rng);
  }
  return vectors;
}

std::vector<int64_t> CustomIds() {
  std::vector<int64_t> ids(kNumRows);
  for (uint32_t i = 0; i < kNumRows; ++i) {
    ids[i] = kIdOffset + i;
  }
  return ids;
}

ArraySeqView MakeBaseView(std::vector<float>* vectors) {
  return ArraySeqView{.data = reinterpret_cast<uint8_t*>(vectors->data()),
                      .dim = kDim,
                      .size = kNumRows,
                      .elem_type = PrimitiveType::kFloatType};
}

PrimitiveSeqView MakeQueryView(std::vector<float>* vectors, uint32_t row) {
  return PrimitiveSeqView{
      .data = reinterpret_cast<uint8_t*>(vectors->data() + static_cast<size_t>(row) * kDim),
      .size = kDim,
      .elem_type = PrimitiveType::kFloatType};
}

void BuildIndex(const IndexMeta& meta, const std::string& path, std::vector<float>* vectors,
                const std::vector<int64_t>& ids) {
  auto builder = IndexFactory::CreateBuilderFromMeta(meta);
  builder->EnableCustomRowId().Open(path).Add({MakeBaseView(vectors)}, ids.data()).Flush().Close();
}

class RejectAllIdFilter : public IdFilter {
 public:
  bool IsMember(idx_t id) const override { return false; }
};

}  // namespace

TEST(FaissIvfSqAnnSearcherTest, FileRoundTripTopKFilterAndRangeSearch) {
  const std::string path = "/tmp/tenann_ivf_sq_l2_test.index";
  auto meta = MakeIvfSqMeta(MetricType::kL2Distance);
  auto vectors = RandomVectors();
  auto ids = CustomIds();
  BuildIndex(meta, path, &vectors, ids);

  auto searcher = AnnSearcherFactory::CreateSearcherFromMeta(meta);
  EXPECT_NE(dynamic_cast<FaissIvfSqAnnSearcher*>(searcher.get()), nullptr);
  searcher->ReadIndex(path);
  ASSERT_NE(searcher->index_ref(), nullptr);
  EXPECT_EQ(searcher->index_ref()->index_type(), IndexType::kFaissIvfSq);

  constexpr uint32_t query_row = 23;
  auto query = MakeQueryView(&vectors, query_row);
  std::vector<int64_t> result_ids(10, -1);
  std::vector<float> result_distances(10);
  searcher->AnnSearch(query, result_ids.size(), result_ids.data(),
                      reinterpret_cast<uint8_t*>(result_distances.data()));
  EXPECT_NE(std::find(result_ids.begin(), result_ids.end(), kIdOffset + query_row),
            result_ids.end());
  EXPECT_TRUE(std::is_sorted(result_distances.begin(), result_distances.end()));

  RejectAllIdFilter reject_all;
  searcher->AnnSearch(query, result_ids.size(), result_ids.data(), &reject_all);
  EXPECT_TRUE(
      std::all_of(result_ids.begin(), result_ids.end(), [](int64_t id) { return id == -1; }));

  std::vector<int64_t> range_ids;
  std::vector<float> range_distances;
  searcher->RangeSearch(query, 1000.0f, 10, AnnSearcher::ResultOrder::kAscending, &range_ids,
                        &range_distances);
  ASSERT_EQ(range_ids.size(), 10u);
  ASSERT_EQ(range_distances.size(), 10u);
  EXPECT_NE(std::find(range_ids.begin(), range_ids.end(), kIdOffset + query_row), range_ids.end());
  EXPECT_TRUE(std::is_sorted(range_distances.begin(), range_distances.end()));

  EXPECT_NO_THROW(searcher->SetSearchParamItem(FaissIvfSqSearchParams::nprobe_key, size_t(4)));
  EXPECT_NO_THROW(searcher->SetSearchParamItem(FaissIvfSqSearchParams::max_codes_key, size_t(100)));
}

TEST(FaissIvfSqAnnSearcherTest, CosineSimilarityNormalizesVectors) {
  auto vectors = RandomVectors();
  auto ids = CustomIds();
  for (const std::string backend : {"l2", "inner_product"}) {
    const std::string path = "/tmp/tenann_ivf_sq_cosine_" + backend + "_test.index";
    auto meta = MakeIvfSqMeta(MetricType::kCosineSimilarity, backend);
    BuildIndex(meta, path, &vectors, ids);

    // The reader must infer the physical metric from the serialized index. A stale writer-only
    // option in reader metadata must not change score conversion or range direction.
    auto reader_meta = meta;
    reader_meta.index_writer_options()["cosine_backend"] = backend == "l2" ? "inner_product" : "l2";
    auto searcher = AnnSearcherFactory::CreateSearcherFromMeta(reader_meta);
    searcher->ReadIndex(path);

    constexpr uint32_t query_row = 47;
    auto query = MakeQueryView(&vectors, query_row);
    std::vector<int64_t> result_ids(10, -1);
    std::vector<float> similarities(10);
    searcher->AnnSearch(query, result_ids.size(), result_ids.data(),
                        reinterpret_cast<uint8_t*>(similarities.data()));
    EXPECT_NE(std::find(result_ids.begin(), result_ids.end(), kIdOffset + query_row),
              result_ids.end())
        << "cosine_backend=" << backend;
    EXPECT_TRUE(std::is_sorted(similarities.rbegin(), similarities.rend()))
        << "cosine_backend=" << backend;
    EXPECT_NEAR(similarities.front(), 1.0f, 0.02f) << "cosine_backend=" << backend;

    std::vector<int64_t> range_ids;
    std::vector<float> range_similarities;
    searcher->RangeSearch(query, 0.5f, 10, AnnSearcher::ResultOrder::kDescending, &range_ids,
                          &range_similarities);
    EXPECT_TRUE(std::is_sorted(range_similarities.rbegin(), range_similarities.rend()))
        << "cosine_backend=" << backend;
    EXPECT_NE(std::find(range_ids.begin(), range_ids.end(), kIdOffset + query_row), range_ids.end())
        << "cosine_backend=" << backend;
  }
}

TEST(FaissIvfSqAnnSearcherTest, InnerProductTopKAndRangeSearch) {
  const std::string path = "/tmp/tenann_ivf_sq_inner_product_test.index";
  auto meta = MakeIvfSqMeta(MetricType::kInnerProduct);
  auto vectors = RandomVectors();
  auto ids = CustomIds();
  constexpr uint32_t query_row = 71;
  std::fill(vectors.begin() + static_cast<size_t>(query_row) * kDim,
            vectors.begin() + static_cast<size_t>(query_row + 1) * kDim, 5.0f);
  BuildIndex(meta, path, &vectors, ids);

  auto searcher = AnnSearcherFactory::CreateSearcherFromMeta(meta);
  searcher->ReadIndex(path);
  auto query = MakeQueryView(&vectors, query_row);

  std::vector<int64_t> result_ids(10, -1);
  std::vector<float> scores(10);
  searcher->AnnSearch(query, result_ids.size(), result_ids.data(),
                      reinterpret_cast<uint8_t*>(scores.data()));
  EXPECT_EQ(result_ids.front(), kIdOffset + query_row);
  EXPECT_TRUE(std::is_sorted(scores.rbegin(), scores.rend()));

  std::vector<int64_t> range_ids;
  std::vector<float> range_scores;
  searcher->RangeSearch(query, 100.0f, 10, AnnSearcher::ResultOrder::kDescending, &range_ids,
                        &range_scores);
  ASSERT_FALSE(range_ids.empty());
  EXPECT_EQ(range_ids.front(), kIdOffset + query_row);
  EXPECT_TRUE(std::is_sorted(range_scores.rbegin(), range_scores.rend()));
}

TEST(FaissIvfSqAnnSearcherTest, RejectsIncompatibleLoadedMetric) {
  const std::string path = "/tmp/tenann_ivf_sq_metric_mismatch_test.index";
  auto writer_meta = MakeIvfSqMeta(MetricType::kL2Distance);
  auto vectors = RandomVectors();
  auto ids = CustomIds();
  BuildIndex(writer_meta, path, &vectors, ids);

  auto reader_meta = writer_meta;
  reader_meta.common_params()["metric_type"] = MetricType::kInnerProduct;
  auto searcher = AnnSearcherFactory::CreateSearcherFromMeta(reader_meta);
  EXPECT_THROW(searcher->ReadIndex(path), Error);
}

}  // namespace tenann
