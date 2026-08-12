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
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <unistd.h>
#include <vector>

#include "faiss/IndexHNSW.h"
#include "faiss/IndexPQ.h"
#include "faiss/impl/DistanceComputer.h"
#include "faiss/utils/distances.h"
#include "gtest/gtest.h"
#include "tenann/builder/faiss_hnsw_index_builder.h"
#include "tenann/common/error.h"
#include "tenann/common/seq_view.h"
#include "tenann/factory/ann_searcher_factory.h"
#include "tenann/factory/index_factory.h"
#include "tenann/index/internal/faiss_index_util.h"
#include "tenann/searcher/faiss_hnsw_ann_searcher.h"
#include "tenann/store/index_meta.h"
#include "tenann/store/index_type.h"

namespace tenann {

namespace {

constexpr uint32_t kDim = 32;
constexpr uint32_t kNb = 2000;
constexpr uint32_t kNq = 5;
constexpr uint32_t kK = 10;

IndexMeta MakeHnswMeta(ScalarQuantizerType q, int m_pq = 0, int nbits_pq = 8,
                       MetricType metric = MetricType::kL2Distance, bool is_vector_normed = false,
                       const char* cosine_backend = nullptr) {
  IndexMeta meta;
  meta.SetMetaVersion(0);
  meta.SetIndexFamily(IndexFamily::kVectorIndex);
  meta.SetIndexType(IndexType::kFaissHnsw);
  meta.common_params()["dim"] = static_cast<int>(kDim);
  meta.common_params()["is_vector_normed"] = is_vector_normed;
  meta.common_params()["metric_type"] = metric;
  meta.index_params()["M"] = 16;
  meta.index_params()["efConstruction"] = 40;
  meta.index_params()["quantizer"] = static_cast<int>(q);
  if (q == ScalarQuantizerType::kPQ) {
    meta.index_params()["m_pq"] = m_pq;
    meta.index_params()["nbits_pq"] = nbits_pq;
  }
  meta.search_params()["efSearch"] = 40;
  meta.search_params()["check_relative_distance"] = true;
  meta.index_writer_options()["write_index_cache"] = false;
  if (cosine_backend != nullptr) {
    meta.index_writer_options()["cosine_backend"] = cosine_backend;
  }
  meta.index_reader_options()["cache_index_file"] = false;
  return meta;
}

std::vector<float> RandomVectors(uint32_t n, uint32_t dim, int seed = 42) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> v(static_cast<size_t>(n) * dim);
  for (auto& x : v) x = dist(rng);
  return v;
}

std::string MakeIndexPath(const std::string& tag) {
  return "/tmp/tenann_hnsw_quantized_test_" + std::to_string(getpid()) + "_" + tag + ".index";
}

}  // namespace

// ---------- GetMinTrainRows ----------

// GetMinTrainRows must work pre-Open() — callers (e.g. StarRocks async build)
// use it to decide whether to build or fall back to brute-force before any
// training work begins.
TEST(FaissHnswQuantizedBuilderTest, MinTrainRowsFlat) {
  auto meta = MakeHnswMeta(ScalarQuantizerType::kFlat);
  FaissHnswIndexBuilder b(meta);
  EXPECT_EQ(b.GetMinTrainRows(), 0u);
}

TEST(FaissHnswQuantizedBuilderTest, MinTrainRowsSq4) {
  auto meta = MakeHnswMeta(ScalarQuantizerType::kSQ4);
  FaissHnswIndexBuilder b(meta);
  EXPECT_EQ(b.GetMinTrainRows(), 1u);
}

TEST(FaissHnswQuantizedBuilderTest, MinTrainRowsSq8) {
  auto meta = MakeHnswMeta(ScalarQuantizerType::kSQ8);
  FaissHnswIndexBuilder b(meta);
  EXPECT_EQ(b.GetMinTrainRows(), 1u);
}

TEST(FaissHnswQuantizedBuilderTest, MinTrainRowsPqDefaultNbits) {
  // nbits_pq=8 -> 256 centroids -> 25600 rows
  auto meta = MakeHnswMeta(ScalarQuantizerType::kPQ, /*m_pq=*/8, /*nbits_pq=*/8);
  FaissHnswIndexBuilder b(meta);
  EXPECT_EQ(b.GetMinTrainRows(), (1u << 8) * 100u);
}

TEST(FaissHnswQuantizedBuilderTest, MinTrainRowsPqSmallNbits) {
  // nbits_pq=4 -> 16 centroids -> 1600 rows
  auto meta = MakeHnswMeta(ScalarQuantizerType::kPQ, /*m_pq=*/8, /*nbits_pq=*/4);
  FaissHnswIndexBuilder b(meta);
  EXPECT_EQ(b.GetMinTrainRows(), (1u << 4) * 100u);
}

TEST(FaissHnswQuantizedBuilderTest, MinTrainRowsPqNbits16RemainsSupported) {
  auto meta = MakeHnswMeta(ScalarQuantizerType::kPQ, /*m_pq=*/8, /*nbits_pq=*/16);
  FaissHnswIndexBuilder b(meta);
  EXPECT_EQ(b.GetMinTrainRows(), (1u << 16) * 100u);
}

// ---------- Build + flush + search round-trip ----------

namespace {

void RunBuildAndSearch(const IndexMeta& meta, const std::string& path_tag,
                       float min_self_recall) {
  auto base = RandomVectors(kNb, kDim, /*seed=*/1);
  std::vector<int64_t> ids(kNb);
  for (uint32_t i = 0; i < kNb; ++i) ids[i] = i;

  ArraySeqView base_view{.data = reinterpret_cast<uint8_t*>(base.data()),
                         .dim = kDim,
                         .size = kNb,
                         .elem_type = PrimitiveType::kFloatType};

  auto path = MakeIndexPath(path_tag);
  auto builder = std::make_unique<FaissHnswIndexBuilder>(meta);
  builder->EnableCustomRowId()
      .Open(path)
      .Add({base_view}, ids.data())
      .Flush()
      .Close();

  // Self-query: pick the first kNq base vectors as queries; the nearest
  // neighbour should be the vector itself for any reasonable quantizer.
  auto searcher = AnnSearcherFactory::CreateSearcherFromMeta(meta);
  searcher->ReadIndex(path);

  std::vector<int64_t> result(kK);
  uint32_t hits = 0;
  for (uint32_t i = 0; i < kNq; ++i) {
    PrimitiveSeqView q{.data = reinterpret_cast<uint8_t*>(base.data() + i * kDim),
                       .size = kDim,
                       .elem_type = PrimitiveType::kFloatType};
    std::fill(result.begin(), result.end(), -1);
    searcher->AnnSearch(q, kK, result.data());
    for (int64_t r : result) {
      if (r == static_cast<int64_t>(i)) {
        ++hits;
        break;
      }
    }
  }
  EXPECT_GE(static_cast<float>(hits) / kNq, min_self_recall)
      << "self-recall too low for tag=" << path_tag;
}

}  // namespace

TEST(FaissHnswQuantizedBuilderTest, BuildAndSearchFlat) {
  // Baseline: HNSWFlat (no quantization, train path bypassed). Self-recall
  // should be perfect.
  RunBuildAndSearch(MakeHnswMeta(ScalarQuantizerType::kFlat), "flat",
                    /*min_self_recall=*/1.0f);
}

TEST(FaissHnswQuantizedBuilderTest, BuildAndSearchSq8) {
  // SQ8 needs train-then-add. Quantization is fine-grained enough that
  // self-query still recovers the vector.
  RunBuildAndSearch(MakeHnswMeta(ScalarQuantizerType::kSQ8), "sq8",
                    /*min_self_recall=*/1.0f);
}

TEST(FaissHnswQuantizedBuilderTest, BuildAndSearchSq4) {
  // SQ4 is coarser; allow the occasional miss.
  RunBuildAndSearch(MakeHnswMeta(ScalarQuantizerType::kSQ4), "sq4",
                    /*min_self_recall=*/0.6f);
}

TEST(FaissHnswQuantizedBuilderTest, BuildAndSearchPq) {
  // dim=32, m_pq=4 (each subvector is 8 floats), nbits_pq=4 -> 16 centroids
  // -> 1600 minimum training rows; we have kNb=2000.
  RunBuildAndSearch(MakeHnswMeta(ScalarQuantizerType::kPQ, /*m_pq=*/4, /*nbits_pq=*/4),
                    "pq4x4", /*min_self_recall=*/0.6f);
}

// ---------- metric x quantizer admission ----------

// SQ4/SQ8 must accept the cosine metric: the `L2Norm,` pre-transform normalizes before
// quantization, so the search-time `cos = 1 - d^2/2` conversion sees exactly the quantized
// L2 distance an l2_distance index already returns.
TEST(FaissHnswQuantizedBuilderTest, Sq8AcceptsCosineMetric) {
  auto meta = MakeHnswMeta(ScalarQuantizerType::kSQ8, 0, 8, MetricType::kCosineSimilarity);
  EXPECT_NO_THROW({ FaissHnswIndexBuilder b(meta); });
}

TEST(FaissHnswQuantizedBuilderTest, Sq4AcceptsCosineMetric) {
  auto meta = MakeHnswMeta(ScalarQuantizerType::kSQ4, 0, 8, MetricType::kCosineSimilarity);
  EXPECT_NO_THROW({ FaissHnswIndexBuilder b(meta); });
}

TEST(FaissHnswQuantizedBuilderTest, Sq8AcceptsCosineMetricPreNormalized) {
  auto meta = MakeHnswMeta(ScalarQuantizerType::kSQ8, 0, 8, MetricType::kCosineSimilarity,
                           /*is_vector_normed=*/true);
  EXPECT_NO_THROW({ FaissHnswIndexBuilder b(meta); });
}

TEST(FaissHnswQuantizedBuilderTest, PqAcceptsCosineMetric) {
  auto meta = MakeHnswMeta(ScalarQuantizerType::kPQ, /*m_pq=*/4, /*nbits_pq=*/4,
                           MetricType::kCosineSimilarity);
  EXPECT_NO_THROW({ FaissHnswIndexBuilder b(meta); });
}

TEST(FaissHnswQuantizedBuilderTest, PqStillAcceptsL2Metric) {
  auto meta =
      MakeHnswMeta(ScalarQuantizerType::kPQ, /*m_pq=*/4, /*nbits_pq=*/4, MetricType::kL2Distance);
  EXPECT_NO_THROW({ FaissHnswIndexBuilder b(meta); });
}

TEST(FaissHnswQuantizedBuilderTest, PqAcceptsInnerProductMetric) {
  auto meta =
      MakeHnswMeta(ScalarQuantizerType::kPQ, /*m_pq=*/4, /*nbits_pq=*/4, MetricType::kInnerProduct);
  EXPECT_NO_THROW({ FaissHnswIndexBuilder b(meta); });
}

TEST(FaissHnswQuantizedBuilderTest, RejectsInvalidCosineBackend) {
  auto meta = MakeHnswMeta(ScalarQuantizerType::kFlat, 0, 8, MetricType::kCosineSimilarity);
  meta.index_writer_options()["cosine_backend"] = "auto";
  EXPECT_THROW({ FaissHnswIndexBuilder b(meta); }, Error);
}

namespace {

void CheckPqSymmetricDistance(const IndexRef& ref, faiss::MetricType metric) {
  auto* raw = static_cast<faiss::Index*>(ref->index_raw());
  auto [_, __, hnsw] = faiss_util::UnpackHnsw(raw);
  ASSERT_NE(hnsw, nullptr);
  ASSERT_EQ(hnsw->storage->metric_type, metric);
  auto* pq = dynamic_cast<const faiss::IndexPQ*>(hnsw->storage);
  ASSERT_NE(pq, nullptr);
  ASSERT_GE(pq->ntotal, 2);
  ASSERT_EQ(pq->pq.sdc_table.size(), pq->pq.M * pq->pq.ksub * pq->pq.ksub);

  std::vector<float> a(kDim), b(kDim);
  pq->reconstruct(0, a.data());
  pq->reconstruct(1, b.data());
  std::unique_ptr<faiss::DistanceComputer> dc(pq->get_distance_computer());
  const float actual = dc->symmetric_dis(0, 1);
  const float expected = metric == faiss::METRIC_INNER_PRODUCT
                             ? faiss::fvec_inner_product(a.data(), b.data(), kDim)
                             : faiss::fvec_L2sqr(a.data(), b.data(), kDim);
  EXPECT_NEAR(actual, expected, 1e-4f * kDim);
}

void RunPqSymmetricDistanceRoundTrip(MetricType logical_metric, const char* cosine_backend,
                                     faiss::MetricType physical_metric, const std::string& tag,
                                     const char* reader_cosine_backend = nullptr) {
  auto meta = MakeHnswMeta(ScalarQuantizerType::kPQ, /*m_pq=*/4, /*nbits_pq=*/4, logical_metric,
                           /*is_vector_normed=*/false, cosine_backend);
  auto base = RandomVectors(kNb, kDim, /*seed=*/19);
  ArraySeqView base_view{.data = reinterpret_cast<uint8_t*>(base.data()),
                         .dim = kDim,
                         .size = kNb,
                         .elem_type = PrimitiveType::kFloatType};

  const auto path = MakeIndexPath("pq_sdc_" + tag);
  auto builder = std::make_unique<FaissHnswIndexBuilder>(meta);
  builder->Open(path).Add({base_view}).Flush();
  CheckPqSymmetricDistance(builder->index_ref(), physical_metric);
  builder->Close();

  // The backend is a writer-only choice. Readers must infer the physical metric from the
  // serialized Faiss index, so even stale or contradictory writer options cannot change how
  // scores are interpreted after a reopen.
  auto reader_meta = meta;
  if (reader_cosine_backend != nullptr) {
    reader_meta.index_writer_options()["cosine_backend"] = reader_cosine_backend;
  }
  auto searcher = AnnSearcherFactory::CreateSearcherFromMeta(reader_meta);
  searcher->ReadIndex(path);
  CheckPqSymmetricDistance(searcher->index_ref(), physical_metric);
}

}  // namespace

TEST(FaissHnswPqSdcTest, L2TableSurvivesRoundTrip) {
  RunPqSymmetricDistanceRoundTrip(MetricType::kL2Distance, nullptr, faiss::METRIC_L2, "l2");
}

TEST(FaissHnswPqSdcTest, InnerProductTableSurvivesRoundTrip) {
  RunPqSymmetricDistanceRoundTrip(MetricType::kInnerProduct, nullptr, faiss::METRIC_INNER_PRODUCT,
                                  "ip");
}

TEST(FaissHnswPqSdcTest, CosineDefaultsToL2Table) {
  RunPqSymmetricDistanceRoundTrip(MetricType::kCosineSimilarity, nullptr, faiss::METRIC_L2,
                                  "cos_l2", "inner_product");
}

TEST(FaissHnswPqSdcTest, IpBackedCosineUsesInnerProductTable) {
  RunPqSymmetricDistanceRoundTrip(MetricType::kCosineSimilarity, "inner_product",
                                  faiss::METRIC_INNER_PRODUCT, "cos_ip", "l2");
}

TEST(FaissHnswPqSdcTest, RejectsLogicalAndPhysicalMetricMismatchOnRead) {
  auto writer_meta = MakeHnswMeta(ScalarQuantizerType::kFlat, 0, 8, MetricType::kL2Distance);
  auto base = RandomVectors(32, kDim, /*seed=*/23);
  ArraySeqView base_view{.data = reinterpret_cast<uint8_t*>(base.data()),
                         .dim = kDim,
                         .size = 32,
                         .elem_type = PrimitiveType::kFloatType};
  const auto path = MakeIndexPath("metric_mismatch");
  auto builder = std::make_unique<FaissHnswIndexBuilder>(writer_meta);
  builder->Open(path).Add({base_view}).Flush().Close();

  auto reader_meta = writer_meta;
  reader_meta.common_params()["metric_type"] = MetricType::kInnerProduct;
  auto searcher = AnnSearcherFactory::CreateSearcherFromMeta(reader_meta);
  EXPECT_THROW(searcher->ReadIndex(path), Error);
}

// ---------- cosine score correctness ----------

namespace {

std::vector<float> RandomSignedVectors(uint32_t n, uint32_t dim, int seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> v(static_cast<size_t>(n) * dim);
  for (auto& x : v) x = dist(rng);
  return v;
}

void NormalizeRows(std::vector<float>& v, uint32_t n, uint32_t dim) {
  faiss::fvec_renorm_L2(dim, n, v.data());
}

float ExactCosine(const float* a, const float* b, uint32_t dim) {
  double dot = 0, na = 0, nb = 0;
  for (uint32_t i = 0; i < dim; ++i) {
    dot += static_cast<double>(a[i]) * b[i];
    na += static_cast<double>(a[i]) * a[i];
    nb += static_cast<double>(b[i]) * b[i];
  }
  return static_cast<float>(dot / (std::sqrt(na) * std::sqrt(nb)));
}

// Asserts the DISTANCES THEMSELVES, not just the ordering. A cosine index that forgets the
// `1 - d/2` conversion (or applies it twice) still returns a perfectly ordered top-k, so an
// order-only assertion would pass while every reported score is wrong.
void RunCosineScoreCheck(const IndexMeta& meta, const std::string& path_tag, float tol) {
  const bool prenormalize = meta.common_params()["is_vector_normed"];
  auto base = RandomSignedVectors(kNb, kDim, /*seed=*/7);
  auto queries = RandomSignedVectors(kNq, kDim, /*seed=*/99);
  if (prenormalize) {
    NormalizeRows(base, kNb, kDim);
    NormalizeRows(queries, kNq, kDim);
  }

  std::vector<int64_t> ids(kNb);
  for (uint32_t i = 0; i < kNb; ++i) ids[i] = i;

  ArraySeqView base_view{.data = reinterpret_cast<uint8_t*>(base.data()),
                         .dim = kDim,
                         .size = kNb,
                         .elem_type = PrimitiveType::kFloatType};

  auto path = MakeIndexPath(path_tag);
  auto builder = std::make_unique<FaissHnswIndexBuilder>(meta);
  builder->EnableCustomRowId().Open(path).Add({base_view}, ids.data()).Flush().Close();

  auto searcher = AnnSearcherFactory::CreateSearcherFromMeta(meta);
  searcher->ReadIndex(path);

  std::vector<int64_t> result_ids(kK);
  std::vector<float> result_dists(kK);
  uint32_t checked = 0;
  for (uint32_t i = 0; i < kNq; ++i) {
    const float* qptr = queries.data() + static_cast<size_t>(i) * kDim;
    PrimitiveSeqView q{.data = reinterpret_cast<uint8_t*>(const_cast<float*>(qptr)),
                       .size = kDim,
                       .elem_type = PrimitiveType::kFloatType};
    std::fill(result_ids.begin(), result_ids.end(), -1);
    std::fill(result_dists.begin(), result_dists.end(), 0.0f);
    searcher->AnnSearch(q, kK, result_ids.data(), reinterpret_cast<uint8_t*>(result_dists.data()));

    for (uint32_t j = 0; j < kK; ++j) {
      const int64_t id = result_ids[j];
      if (id < 0) continue;
      ASSERT_LT(id, static_cast<int64_t>(kNb));
      const float expect = ExactCosine(qptr, base.data() + static_cast<size_t>(id) * kDim, kDim);
      // A cosine similarity must stay inside [-1, 1] up to quantization slack.
      EXPECT_GE(result_dists[j], -1.0f - tol) << "tag=" << path_tag << " j=" << j;
      EXPECT_LE(result_dists[j], 1.0f + tol) << "tag=" << path_tag << " j=" << j;
      EXPECT_NEAR(result_dists[j], expect, tol)
          << "tag=" << path_tag << " query=" << i << " rank=" << j << " id=" << id;
      ++checked;
    }
  }
  EXPECT_GT(checked, 0u) << "no results returned for tag=" << path_tag;
}

}  // namespace

// Baseline: without quantization the identity cos = 1 - d^2/2 is exact (delta = 0), so the
// returned score should match the exact cosine to within float noise.
TEST(FaissHnswQuantizedBuilderTest, CosineFlatScoreIsExact) {
  RunCosineScoreCheck(MakeHnswMeta(ScalarQuantizerType::kFlat, 0, 8, MetricType::kCosineSimilarity),
                      "cos_flat", /*tol=*/1e-3f);
}

// SQ8: the reported cosine carries the quantization error, which is exactly -1/2 of the L2
// error already accepted for metric_type=l2_distance. Tolerance is loose enough to absorb
// that but far tighter than a missing/duplicated conversion would produce.
TEST(FaissHnswQuantizedBuilderTest, CosineSq8ScoreWithinQuantizationError) {
  RunCosineScoreCheck(MakeHnswMeta(ScalarQuantizerType::kSQ8, 0, 8, MetricType::kCosineSimilarity),
                      "cos_sq8", /*tol=*/0.02f);
}

// SQ4 is far coarser (16 levels per dimension), so the score tolerance widens accordingly.
TEST(FaissHnswQuantizedBuilderTest, CosineSq4ScoreWithinQuantizationError) {
  RunCosineScoreCheck(MakeHnswMeta(ScalarQuantizerType::kSQ4, 0, 8, MetricType::kCosineSimilarity),
                      "cos_sq4", /*tol=*/0.2f);
}

// is_vector_normed=true skips the L2Norm pre-transform; the caller guarantees unit-norm
// input on BOTH sides. Scores must still be correct.
TEST(FaissHnswQuantizedBuilderTest, CosineSq8ScorePreNormalizedInput) {
  RunCosineScoreCheck(MakeHnswMeta(ScalarQuantizerType::kSQ8, 0, 8, MetricType::kCosineSimilarity,
                                   /*is_vector_normed=*/true),
                      "cos_sq8_normed", /*tol=*/0.02f);
}

// ---------- cosine + SQ end-to-end recall ----------

TEST(FaissHnswQuantizedBuilderTest, BuildAndSearchCosineFlat) {
  RunBuildAndSearch(MakeHnswMeta(ScalarQuantizerType::kFlat, 0, 8, MetricType::kCosineSimilarity),
                    "cos_flat_recall", /*min_self_recall=*/1.0f);
}

TEST(FaissHnswQuantizedBuilderTest, BuildAndSearchCosineSq8) {
  RunBuildAndSearch(MakeHnswMeta(ScalarQuantizerType::kSQ8, 0, 8, MetricType::kCosineSimilarity),
                    "cos_sq8_recall", /*min_self_recall=*/1.0f);
}

TEST(FaissHnswQuantizedBuilderTest, BuildAndSearchCosineSq4) {
  RunBuildAndSearch(MakeHnswMeta(ScalarQuantizerType::kSQ4, 0, 8, MetricType::kCosineSimilarity),
                    "cos_sq4_recall", /*min_self_recall=*/0.6f);
}

// ---------- inner product ----------
//
// HNSW traversal is written for "smaller is closer", so a similarity metric is served by
// wrapping the distance computer in faiss' NegativeDistanceComputer and flipping the sign
// back once the search finishes. Every assertion below checks the RETURNED SCORE, not just
// the ordering: a missing sign flip leaves the ranking perfectly intact while negating every
// value, which an order-only test cannot see.

namespace {

float ExactDot(const float* a, const float* b, uint32_t dim) {
  double dot = 0;
  for (uint32_t i = 0; i < dim; ++i) dot += static_cast<double>(a[i]) * b[i];
  return static_cast<float>(dot);
}

// Vectors the index was built from, the queries to probe it with, and a loaded searcher.
struct Fixture {
  std::vector<float> base;
  std::vector<float> queries;
  std::shared_ptr<AnnSearcher> searcher;
};

Fixture BuildIpIndex(const IndexMeta& meta, const std::string& path_tag) {
  Fixture f;
  f.base = RandomSignedVectors(kNb, kDim, /*seed=*/11);
  f.queries = RandomSignedVectors(kNq, kDim, /*seed=*/23);

  std::vector<int64_t> ids(kNb);
  for (uint32_t i = 0; i < kNb; ++i) ids[i] = i;

  ArraySeqView base_view{.data = reinterpret_cast<uint8_t*>(f.base.data()),
                         .dim = kDim,
                         .size = kNb,
                         .elem_type = PrimitiveType::kFloatType};

  auto path = MakeIndexPath(path_tag);
  auto builder = std::make_unique<FaissHnswIndexBuilder>(meta);
  builder->EnableCustomRowId().Open(path).Add({base_view}, ids.data()).Flush().Close();

  f.searcher = AnnSearcherFactory::CreateSearcherFromMeta(meta);
  f.searcher->ReadIndex(path);
  return f;
}

PrimitiveSeqView ViewOf(const float* p) {
  return PrimitiveSeqView{.data = reinterpret_cast<uint8_t*>(const_cast<float*>(p)),
                          .size = kDim,
                          .elem_type = PrimitiveType::kFloatType};
}

void RunIpTopKCheck(const IndexMeta& meta, const std::string& tag, float tol,
                    bool compare_reconstructed_codes = false) {
  auto f = BuildIpIndex(meta, tag);
  std::vector<int64_t> rid(kK);
  std::vector<float> rd(kK);
  std::vector<float> reconstructed(kDim);
  const faiss::Index* storage = nullptr;
  if (compare_reconstructed_codes) {
    auto* raw = static_cast<faiss::Index*>(f.searcher->index_ref()->index_raw());
    auto [_, __, hnsw] = faiss_util::UnpackHnsw(raw);
    ASSERT_NE(hnsw, nullptr);
    storage = hnsw->storage;
    ASSERT_NE(storage, nullptr);
  }

  for (uint32_t i = 0; i < kNq; ++i) {
    const float* q = f.queries.data() + static_cast<size_t>(i) * kDim;
    std::fill(rid.begin(), rid.end(), -1);
    f.searcher->AnnSearch(ViewOf(f.queries.data() + static_cast<size_t>(i) * kDim), kK, rid.data(),
                          reinterpret_cast<uint8_t*>(rd.data()));
    float prev = std::numeric_limits<float>::max();
    for (uint32_t j = 0; j < kK; ++j) {
      if (rid[j] < 0) continue;
      ASSERT_LT(rid[j], static_cast<int64_t>(kNb));
      const float* base = f.base.data() + static_cast<size_t>(rid[j]) * kDim;
      if (storage != nullptr) {
        storage->reconstruct(rid[j], reconstructed.data());
        base = reconstructed.data();
      }
      const float expect = ExactDot(q, base, kDim);
      EXPECT_NEAR(rd[j], expect, tol) << "tag=" << tag << " q=" << i << " rank=" << j;
      // A similarity must come back descending.
      EXPECT_LE(rd[j], prev + tol) << "tag=" << tag << " not descending at rank " << j;
      prev = rd[j];
    }
  }
}

void RunIpRangeSearchCheck(const IndexMeta& meta, const std::string& tag, int64_t limit,
                           float threshold, float tol) {
  auto f = BuildIpIndex(meta, tag);
  uint32_t total = 0;

  for (uint32_t i = 0; i < kNq; ++i) {
    const float* q = f.queries.data() + static_cast<size_t>(i) * kDim;
    std::vector<int64_t> ids;
    std::vector<float> dists;
    f.searcher->RangeSearch(ViewOf(f.queries.data() + static_cast<size_t>(i) * kDim), threshold,
                            limit, AnnSearcher::ResultOrder::kDescending, &ids, &dists);
    ASSERT_EQ(ids.size(), dists.size());

    float prev = std::numeric_limits<float>::max();
    for (size_t j = 0; j < ids.size(); ++j) {
      ASSERT_GE(ids[j], 0);
      ASSERT_LT(ids[j], static_cast<int64_t>(kNb));
      const float expect = ExactDot(q, f.base.data() + static_cast<size_t>(ids[j]) * kDim, kDim);
      // A missing sign flip would report -expect, blowing past both of these.
      EXPECT_NEAR(dists[j], expect, tol) << "tag=" << tag << " q=" << i << " j=" << j;
      // `threshold` is a LOWER bound for a similarity metric.
      EXPECT_GE(dists[j], threshold - tol) << "tag=" << tag << " q=" << i << " j=" << j;
      EXPECT_LE(dists[j], prev + tol) << "tag=" << tag << " not descending at " << j;
      prev = dists[j];
      ++total;
    }
  }
  EXPECT_GT(total, 0u) << "range search returned nothing for tag=" << tag
                       << "; threshold may be too high for the fixture";
}

}  // namespace

TEST(FaissHnswIpTest, TopKFlatScoresAreExactDotProducts) {
  RunIpTopKCheck(MakeHnswMeta(ScalarQuantizerType::kFlat, 0, 8, MetricType::kInnerProduct),
                 "ip_flat_topk", /*tol=*/1e-2f);
}

TEST(FaissHnswIpTest, TopKSq8ScoresTrackDotProducts) {
  RunIpTopKCheck(MakeHnswMeta(ScalarQuantizerType::kSQ8, 0, 8, MetricType::kInnerProduct),
                 "ip_sq8_topk", /*tol=*/0.5f);
}

TEST(FaissHnswIpTest, TopKPqScoresTrackDotProducts) {
  RunIpTopKCheck(
      MakeHnswMeta(ScalarQuantizerType::kPQ, /*m_pq=*/4, /*nbits_pq=*/4, MetricType::kInnerProduct),
      "ip_pq_topk", /*tol=*/1e-3f, /*compare_reconstructed_codes=*/true);
}

// limit > 0 goes through index.search(), which faiss has already sign-corrected; that branch
// only has to keep the prefix clearing the threshold, comparing with >= rather than <=.
TEST(FaissHnswIpTest, RangeSearchWithLimitHonoursLowerBound) {
  RunIpRangeSearchCheck(MakeHnswMeta(ScalarQuantizerType::kFlat, 0, 8, MetricType::kInnerProduct),
                        "ip_flat_range_limited", /*limit=*/kK, /*threshold=*/10.0f, /*tol=*/1e-2f);
}

// limit <= 0 is the hand-rolled branch: it drives NegativeDistanceComputer directly, so it
// must negate the radius on the way in AND negate the scores on the way out. This is what
// faiss upstream does in IndexHNSW::range_search and what tenann previously refused to do.
TEST(FaissHnswIpTest, RangeSearchUnlimitedHonoursLowerBound) {
  RunIpRangeSearchCheck(MakeHnswMeta(ScalarQuantizerType::kFlat, 0, 8, MetricType::kInnerProduct),
                        "ip_flat_range_unlimited", /*limit=*/-1, /*threshold=*/10.0f,
                        /*tol=*/1e-2f);
}

TEST(FaissHnswIpTest, RangeSearchUnlimitedSq8) {
  RunIpRangeSearchCheck(MakeHnswMeta(ScalarQuantizerType::kSQ8, 0, 8, MetricType::kInnerProduct),
                        "ip_sq8_range_unlimited", /*limit=*/-1, /*threshold=*/10.0f, /*tol=*/0.5f);
}

// Ascending order makes no sense for a similarity metric and must be rejected outright.
TEST(FaissHnswIpTest, RangeSearchRejectsAscendingOrder) {
  auto meta = MakeHnswMeta(ScalarQuantizerType::kFlat, 0, 8, MetricType::kInnerProduct);
  auto f = BuildIpIndex(meta, "ip_flat_order_reject");
  std::vector<int64_t> ids;
  std::vector<float> dists;
  EXPECT_THROW(
      {
        f.searcher->RangeSearch(ViewOf(f.queries.data()), 10.0f, -1,
                                AnnSearcher::ResultOrder::kAscending, &ids, &dists);
      },
      Error);
}

// ---------- IVFPQ + inner product ----------
//
// IVFPQ range search runs through tenann's own custom_range_search_preassigned. Its optional
// confidence widening is L2-only (it takes sqrtf() of a squared distance and bounds the true
// distance from below via ||a - a_hat||), so a similarity metric must fall through to the plain
// metric-specialized C::cmp path. On top of that the searcher's result heap keeps the "best"
// entries, which means largest-first for a similarity and smallest-first for a distance.

namespace {

IndexMeta MakeIvfPqMeta(MetricType metric, const char* cosine_backend = nullptr) {
  IndexMeta meta;
  meta.SetMetaVersion(0);
  meta.SetIndexFamily(IndexFamily::kVectorIndex);
  meta.SetIndexType(IndexType::kFaissIvfPq);
  meta.common_params()["dim"] = static_cast<int>(kDim);
  meta.common_params()["is_vector_normed"] = false;
  meta.common_params()["metric_type"] = metric;
  meta.index_params()["nlist"] = 16;
  meta.index_params()["M"] = 4;
  meta.index_params()["nbits"] = 8;
  meta.search_params()["nprobe"] = 16;
  meta.index_writer_options()["write_index_cache"] = false;
  if (cosine_backend != nullptr) {
    meta.index_writer_options()["cosine_backend"] = cosine_backend;
  }
  meta.index_reader_options()["cache_index_file"] = false;
  return meta;
}

}  // namespace

TEST(FaissIvfPqIpTest, RangeSearchHonoursLowerBoundAndScores) {
  auto meta = MakeIvfPqMeta(MetricType::kInnerProduct);
  auto base = RandomSignedVectors(kNb, kDim, /*seed=*/31);
  auto queries = RandomSignedVectors(kNq, kDim, /*seed=*/37);

  std::vector<int64_t> ids(kNb);
  for (uint32_t i = 0; i < kNb; ++i) ids[i] = i;
  ArraySeqView base_view{.data = reinterpret_cast<uint8_t*>(base.data()),
                         .dim = kDim,
                         .size = kNb,
                         .elem_type = PrimitiveType::kFloatType};

  auto path = MakeIndexPath("ivfpq_ip_range");
  auto builder = IndexFactory::CreateBuilderFromMeta(meta);
  builder->EnableCustomRowId().Open(path).Add({base_view}, ids.data()).Flush().Close();

  auto searcher = AnnSearcherFactory::CreateSearcherFromMeta(meta);
  searcher->ReadIndex(path);

  const float threshold = 8.0f;
  uint32_t total = 0;
  // PQ reconstruction on 32-d gaussian noise is coarse (M=4 sub-quantizers, 2000 training rows),
  // so an individual score can sit far from the true dot product -- Cauchy-Schwarz alone allows
  // |q.a - q.a_hat| <= ||q|| * ||a - a_hat||, which is over 10 units here. Asserting a per-result
  // absolute tolerance would therefore be either flaky or meaningless. Instead the per-result
  // assertions below are the two EXACT ones, and agreement with the true dot product is checked
  // in aggregate via correlation, which a sign error inverts wholesale.
  double correlation = 0;

  for (uint32_t i = 0; i < kNq; ++i) {
    const float* q = queries.data() + static_cast<size_t>(i) * kDim;
    PrimitiveSeqView qv{.data = reinterpret_cast<uint8_t*>(const_cast<float*>(q)),
                        .size = kDim,
                        .elem_type = PrimitiveType::kFloatType};
    std::vector<int64_t> rid;
    std::vector<float> rd;
    searcher->RangeSearch(qv, threshold, /*limit=*/-1, AnnSearcher::ResultOrder::kDescending, &rid,
                          &rd);
    ASSERT_EQ(rid.size(), rd.size());

    float prev = std::numeric_limits<float>::max();
    for (size_t j = 0; j < rid.size(); ++j) {
      ASSERT_GE(rid[j], 0);
      ASSERT_LT(rid[j], static_cast<int64_t>(kNb));
      const float expect = ExactDot(q, base.data() + static_cast<size_t>(rid[j]) * kDim, kDim);
      // EXACT: the index filtered on the very value it reports, so every result must clear the
      // threshold with no slack. This is what breaks if the similarity branch is missing.
      EXPECT_GE(rd[j], threshold - 1e-3f) << "q=" << i << " j=" << j << " score=" << rd[j];
      // EXACT: a similarity must come back largest-first.
      EXPECT_LE(rd[j], prev + 1e-3f) << "not descending at " << j;
      correlation += static_cast<double>(rd[j]) * expect;
      prev = rd[j];
      ++total;
    }
  }
  EXPECT_GT(total, 0u) << "ivfpq ip range search returned nothing";
  // Reported scores must agree in sign with the true dot products. A missing similarity branch
  // would report negated values, flipping every term and driving this negative.
  EXPECT_GT(correlation, 0.0) << "reported scores do not correlate with true dot products";
}

TEST(FaissIvfPqIpTest, RangeSearchRejectsAscendingOrder) {
  auto meta = MakeIvfPqMeta(MetricType::kInnerProduct);
  auto base = RandomSignedVectors(kNb, kDim, /*seed=*/31);
  std::vector<int64_t> ids(kNb);
  for (uint32_t i = 0; i < kNb; ++i) ids[i] = i;
  ArraySeqView base_view{.data = reinterpret_cast<uint8_t*>(base.data()),
                         .dim = kDim,
                         .size = kNb,
                         .elem_type = PrimitiveType::kFloatType};
  auto path = MakeIndexPath("ivfpq_ip_order_reject");
  auto builder = IndexFactory::CreateBuilderFromMeta(meta);
  builder->EnableCustomRowId().Open(path).Add({base_view}, ids.data()).Flush().Close();

  auto searcher = AnnSearcherFactory::CreateSearcherFromMeta(meta);
  searcher->ReadIndex(path);

  PrimitiveSeqView qv{.data = reinterpret_cast<uint8_t*>(base.data()),
                      .size = kDim,
                      .elem_type = PrimitiveType::kFloatType};
  std::vector<int64_t> rid;
  std::vector<float> rd;
  EXPECT_THROW(
      { searcher->RangeSearch(qv, 8.0f, -1, AnnSearcher::ResultOrder::kAscending, &rid, &rd); },
      Error);
}

// ---------- cosine range search, and the guarantees the score carries ----------
//
// An opt-in IP-backed cosine index takes a different range-search route from the legacy L2-backed
// form: the threshold is used as-is instead of being converted to an L2 bound, and it acts as a
// lower bound.

namespace {

Fixture BuildCosIndex(ScalarQuantizerType q, bool normed, const std::string& tag,
                      const char* cosine_backend = "inner_product") {
  Fixture f;
  const int m_pq = q == ScalarQuantizerType::kPQ ? 4 : 0;
  const int nbits_pq = q == ScalarQuantizerType::kPQ ? 4 : 8;
  auto meta =
      MakeHnswMeta(q, m_pq, nbits_pq, MetricType::kCosineSimilarity, normed, cosine_backend);
  f.base = RandomSignedVectors(kNb, kDim, /*seed=*/11);
  f.queries = RandomSignedVectors(kNq, kDim, /*seed=*/23);
  if (normed) {
    NormalizeRows(f.base, kNb, kDim);
    NormalizeRows(f.queries, kNq, kDim);
  }
  std::vector<int64_t> ids(kNb);
  for (uint32_t i = 0; i < kNb; ++i) ids[i] = i;
  ArraySeqView bv{.data = reinterpret_cast<uint8_t*>(f.base.data()),
                  .dim = kDim,
                  .size = kNb,
                  .elem_type = PrimitiveType::kFloatType};
  auto path = MakeIndexPath("cosrange_" + tag + "_" + cosine_backend);
  auto builder = std::make_unique<FaissHnswIndexBuilder>(meta);
  builder->EnableCustomRowId().Open(path).Add({bv}, ids.data()).Flush().Close();
  f.searcher = AnnSearcherFactory::CreateSearcherFromMeta(meta);
  f.searcher->ReadIndex(path);
  return f;
}

void RunCosRangeCheck(ScalarQuantizerType q, bool normed, const std::string& tag, int64_t limit,
                      float threshold, const char* cosine_backend = "inner_product") {
  auto f = BuildCosIndex(q, normed, tag + (limit > 0 ? "_lim" : "_unlim"), cosine_backend);
  uint32_t total = 0;
  for (uint32_t i = 0; i < kNq; ++i) {
    std::vector<int64_t> ids;
    std::vector<float> scores;
    f.searcher->RangeSearch(ViewOf(f.queries.data() + static_cast<size_t>(i) * kDim), threshold,
                            limit, AnnSearcher::ResultOrder::kDescending, &ids, &scores);
    ASSERT_EQ(ids.size(), scores.size());
    if (limit > 0) ASSERT_LE(static_cast<int64_t>(ids.size()), limit);

    float prev = std::numeric_limits<float>::max();
    for (size_t j = 0; j < ids.size(); ++j) {
      // The threshold is a LOWER bound: an L2-backed index reaches that by converting it into an
      // upper bound on distance, an IP-backed one by comparing against it directly. Both must end
      // up here.
      EXPECT_GE(scores[j], threshold - 1e-3f) << tag << " q=" << i << " j=" << j;
      EXPECT_LE(scores[j], prev + 1e-3f) << tag << " not descending at " << j;
      // Whatever the backing metric, the output is a cosine and must look like one.
      EXPECT_LE(scores[j], 1.0f) << tag << " score above 1";
      EXPECT_GE(scores[j], -1.0f) << tag << " score below -1";
      prev = scores[j];
      ++total;
    }
  }
  EXPECT_GT(total, 0u) << "no results for " << tag << "; threshold too high for the fixture?";
}

}  // namespace

TEST(FaissHnswCosineRangeTest, Sq8UnnormedWithLimit) {
  RunCosRangeCheck(ScalarQuantizerType::kSQ8, false, "sq8_unnormed", /*limit=*/kK, 0.3f);
}

TEST(FaissHnswCosineRangeTest, Sq8UnnormedUnlimited) {
  RunCosRangeCheck(ScalarQuantizerType::kSQ8, false, "sq8_unnormed", /*limit=*/-1, 0.3f);
}

TEST(FaissHnswCosineRangeTest, Sq8NormedUnlimited) {
  RunCosRangeCheck(ScalarQuantizerType::kSQ8, true, "sq8_normed", /*limit=*/-1, 0.3f);
}

TEST(FaissHnswCosineRangeTest, Sq4UnnormedUnlimited) {
  RunCosRangeCheck(ScalarQuantizerType::kSQ4, false, "sq4_unnormed", /*limit=*/-1, 0.3f);
}

TEST(FaissHnswCosineRangeTest, PqUnnormedUnlimited) {
  RunCosRangeCheck(ScalarQuantizerType::kPQ, false, "pq_unnormed", /*limit=*/-1, 0.3f);
}

// The default L2-backed form must remain readable and keep its threshold conversion.
TEST(FaissHnswCosineRangeTest, LegacyL2BackedFlatUnnormedUnlimited) {
  RunCosRangeCheck(ScalarQuantizerType::kFlat, false, "flat_unnormed", /*limit=*/-1, 0.3f,
                   /*cosine_backend=*/"l2");
}

// A negative threshold is legal for a similarity and simply admits more.
TEST(FaissHnswCosineRangeTest, NegativeThresholdAdmitsMore) {
  auto f = BuildCosIndex(ScalarQuantizerType::kSQ8, false, "sq8_negthresh");
  auto qv = ViewOf(f.queries.data());
  std::vector<int64_t> loose_ids, tight_ids;
  std::vector<float> loose, tight;
  f.searcher->RangeSearch(qv, -0.5f, -1, AnnSearcher::ResultOrder::kDescending, &loose_ids, &loose);
  f.searcher->RangeSearch(qv, 0.3f, -1, AnnSearcher::ResultOrder::kDescending, &tight_ids, &tight);
  EXPECT_GE(loose_ids.size(), tight_ids.size());
  for (float s : loose) EXPECT_GE(s, -0.5f - 1e-3f);
}

// ---------- score guarantees ----------

// Quantization leaves ||a_hat|| != 1, so an inner-product-backed cosine overshoots 1 on a
// near-duplicate unless the score is clamped. Self-queries are exactly that case: without the
// clamp roughly half of these come back above 1 (up to 1.043 for SQ4).
TEST(FaissHnswCosineScoreTest, SelfQueryNeverExceedsOne) {
  for (auto q :
       {ScalarQuantizerType::kFlat, ScalarQuantizerType::kSQ8, ScalarQuantizerType::kSQ4}) {
    auto f = BuildCosIndex(q, /*normed=*/true, "selfq");
    std::vector<int64_t> ids(kK);
    std::vector<float> scores(kK);
    for (uint32_t i = 0; i < 50; ++i) {
      f.searcher->AnnSearch(ViewOf(f.base.data() + static_cast<size_t>(i) * kDim), kK, ids.data(),
                            reinterpret_cast<uint8_t*>(scores.data()));
      for (uint32_t j = 0; j < kK; ++j) {
        EXPECT_LE(scores[j], 1.0f) << "quantizer=" << static_cast<int>(q) << " row=" << i;
        EXPECT_GE(scores[j], -1.0f) << "quantizer=" << static_cast<int>(q) << " row=" << i;
      }
    }
  }
}

// Cosine is defined on directions, so scaling the query must not move the score. An index with a
// normalizing pre-transform gets this for free; one built with is_vector_normed=true does not, and
// relies on the searcher normalizing the query itself.
TEST(FaissHnswCosineScoreTest, ScoreIsInvariantToQueryScale) {
  for (bool normed : {false, true}) {
    for (auto q : {ScalarQuantizerType::kFlat, ScalarQuantizerType::kSQ8}) {
      auto f = BuildCosIndex(q, normed, std::string("scale_") + (normed ? "n" : "u"));
      std::vector<int64_t> ids1(kK), ids2(kK);
      std::vector<float> s1(kK), s2(kK);
      const float* q0 = f.queries.data();
      f.searcher->AnnSearch(ViewOf(q0), kK, ids1.data(), reinterpret_cast<uint8_t*>(s1.data()));

      std::vector<float> scaled(q0, q0 + kDim);
      for (auto& v : scaled) v *= 7.0f;
      f.searcher->AnnSearch(ViewOf(scaled.data()), kK, ids2.data(),
                            reinterpret_cast<uint8_t*>(s2.data()));

      for (uint32_t j = 0; j < kK; ++j) {
        EXPECT_EQ(ids1[j], ids2[j]) << "normed=" << normed << " rank=" << j;
        EXPECT_NEAR(s1[j], s2[j], 1e-4f) << "normed=" << normed << " rank=" << j;
      }
    }
  }
}

// A zero query has no direction. It must not produce NaNs, and must stay inside the cosine range.
TEST(FaissHnswCosineScoreTest, ZeroQueryStaysFinite) {
  for (auto q : {ScalarQuantizerType::kFlat, ScalarQuantizerType::kSQ8}) {
    auto f = BuildCosIndex(q, /*normed=*/true, "zeroq");
    std::vector<float> zero(kDim, 0.0f);
    std::vector<int64_t> ids(kK);
    std::vector<float> scores(kK);
    f.searcher->AnnSearch(ViewOf(zero.data()), kK, ids.data(),
                          reinterpret_cast<uint8_t*>(scores.data()));
    for (uint32_t j = 0; j < kK; ++j) {
      EXPECT_TRUE(std::isfinite(scores[j])) << "quantizer=" << static_cast<int>(q);
      EXPECT_LE(scores[j], 1.0f);
      EXPECT_GE(scores[j], -1.0f);
    }
  }
}

class ScoreFinalizerForTest : public FaissHnswAnnSearcher {
 public:
  explicit ScoreFinalizerForTest(const IndexMeta& meta) : FaissHnswAnnSearcher(meta) {}

  using AnnSearcher::FinalizeScores;
  using AnnSearcher::PrepareCosineRange;
};

TEST(FaissHnswCosineRangeTest, ValidatesThresholdForEveryPhysicalMetric) {
  auto meta = MakeHnswMeta(ScalarQuantizerType::kFlat, 0, 8, MetricType::kCosineSimilarity);
  ScoreFinalizerForTest searcher(meta);

  EXPECT_FLOAT_EQ(searcher.PrepareCosineRange(-1.0f, faiss::METRIC_L2), 4.0f);
  EXPECT_FLOAT_EQ(searcher.PrepareCosineRange(1.0f, faiss::METRIC_INNER_PRODUCT), 1.0f);
  for (auto physical_metric : {faiss::METRIC_L2, faiss::METRIC_INNER_PRODUCT}) {
    EXPECT_THROW(searcher.PrepareCosineRange(-1.01f, physical_metric), Error);
    EXPECT_THROW(searcher.PrepareCosineRange(1.01f, physical_metric), Error);
    EXPECT_THROW(
        searcher.PrepareCosineRange(std::numeric_limits<float>::quiet_NaN(), physical_metric),
        Error);
  }
}

TEST(FaissHnswCosineScoreTest, PaddingKeepsFaissSentinels) {
  auto meta = MakeHnswMeta(ScalarQuantizerType::kFlat, 0, 8, MetricType::kCosineSimilarity);
  ScoreFinalizerForTest searcher(meta);
  const int64_t ids[] = {7, -1};

  float l2_scores[] = {0.0f, std::numeric_limits<float>::infinity()};
  searcher.FinalizeScores(ids, l2_scores, 2, faiss::METRIC_L2);
  EXPECT_FLOAT_EQ(l2_scores[0], 1.0f);
  EXPECT_EQ(l2_scores[1], std::numeric_limits<float>::infinity());

  float ip_scores[] = {2.0f, -std::numeric_limits<float>::infinity()};
  searcher.FinalizeScores(ids, ip_scores, 2, faiss::METRIC_INNER_PRODUCT);
  EXPECT_FLOAT_EQ(ip_scores[0], 1.0f);
  EXPECT_EQ(ip_scores[1], -std::numeric_limits<float>::infinity());
}

// ---------- IVFPQ carries the same cosine guarantees as HNSW ----------
//
// Query normalization and score clamping live on AnnSearcher, not on one subclass, so both
// searchers get them. Before that they were HNSW-only and an IVFPQ cosine index built with
// is_vector_normed=true scored a 7x-scaled query at -18.5 instead of ~1.

TEST(FaissIvfPqCosineTest, ScoreIsInvariantToQueryScale) {
  for (bool normed : {false, true}) {
    for (const char* backend : {"l2", "inner_product"}) {
      auto meta = MakeIvfPqMeta(MetricType::kCosineSimilarity, backend);
      meta.common_params()["is_vector_normed"] = normed;
      auto base = RandomSignedVectors(kNb, kDim, /*seed=*/31);
      NormalizeRows(base, kNb, kDim);
      std::vector<int64_t> ids(kNb);
      for (uint32_t i = 0; i < kNb; ++i) ids[i] = i;
      ArraySeqView bv{.data = reinterpret_cast<uint8_t*>(base.data()),
                      .dim = kDim,
                      .size = kNb,
                      .elem_type = PrimitiveType::kFloatType};
      auto path = MakeIndexPath(std::string("ivfpq_cos_") + backend + (normed ? "_n" : "_u"));
      auto builder = IndexFactory::CreateBuilderFromMeta(meta);
      builder->EnableCustomRowId().Open(path).Add({bv}, ids.data()).Flush().Close();
      auto reader_meta = meta;
      reader_meta.index_writer_options()["cosine_backend"] =
          std::string(backend) == "l2" ? "inner_product" : "l2";
      auto searcher = AnnSearcherFactory::CreateSearcherFromMeta(reader_meta);
      searcher->ReadIndex(path);

      std::vector<int64_t> ids1(kK), ids2(kK);
      std::vector<float> s1(kK), s2(kK);
      searcher->AnnSearch(ViewOf(base.data()), kK, ids1.data(),
                          reinterpret_cast<uint8_t*>(s1.data()));

      std::vector<float> scaled(base.begin(), base.begin() + kDim);
      for (auto& v : scaled) v *= 7.0f;
      searcher->AnnSearch(ViewOf(scaled.data()), kK, ids2.data(),
                          reinterpret_cast<uint8_t*>(s2.data()));

      for (uint32_t j = 0; j < kK; ++j) {
        EXPECT_EQ(ids1[j], ids2[j])
            << "backend=" << backend << " normed=" << normed << " rank=" << j;
        EXPECT_NEAR(s1[j], s2[j], 1e-4f)
            << "backend=" << backend << " normed=" << normed << " rank=" << j;
        EXPECT_LE(s1[j], 1.0f) << "backend=" << backend << " normed=" << normed;
        EXPECT_GE(s1[j], -1.0f) << "backend=" << backend << " normed=" << normed;
      }
    }
  }
}

TEST(FaissIvfPqCosineTest, IpBackedRangeUsesCosineLowerBound) {
  auto meta = MakeIvfPqMeta(MetricType::kCosineSimilarity, "inner_product");
  auto base = RandomSignedVectors(kNb, kDim, /*seed=*/43);
  std::vector<int64_t> ids(kNb);
  std::iota(ids.begin(), ids.end(), 0);
  ArraySeqView bv{.data = reinterpret_cast<uint8_t*>(base.data()),
                  .dim = kDim,
                  .size = kNb,
                  .elem_type = PrimitiveType::kFloatType};
  const auto path = MakeIndexPath("ivfpq_cos_ip_range");
  auto builder = IndexFactory::CreateBuilderFromMeta(meta);
  builder->EnableCustomRowId().Open(path).Add({bv}, ids.data()).Flush().Close();
  auto searcher = AnnSearcherFactory::CreateSearcherFromMeta(meta);
  searcher->ReadIndex(path);

  std::vector<int64_t> result_ids;
  std::vector<float> scores;
  searcher->RangeSearch(ViewOf(base.data()), 0.3f, /*limit=*/20,
                        AnnSearcher::ResultOrder::kDescending, &result_ids, &scores);
  ASSERT_FALSE(scores.empty());
  ASSERT_EQ(result_ids.size(), scores.size());
  for (size_t i = 0; i < scores.size(); ++i) {
    EXPECT_GE(scores[i], 0.3f);
    EXPECT_LE(scores[i], 1.0f);
    EXPECT_GE(scores[i], -1.0f);
    if (i > 0) {
      EXPECT_LE(scores[i], scores[i - 1]);
    }
  }
}

}  // namespace tenann
