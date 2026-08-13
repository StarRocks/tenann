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

#include <limits>
#include <memory>

#include "faiss/IndexHNSW.h"
#include "faiss/IndexIDMap.h"
#include "faiss/IndexPQ.h"
#include "faiss/IndexPreTransform.h"
#include "faiss/index_factory.h"
#include "gtest/gtest.h"
#include "tenann/index/internal/faiss_index_util.h"
#include "tenann/index/internal/metric_util.h"
#include "tenann/index/parameters.h"
#include "tenann/store/index_type.h"

namespace tenann {

namespace {

VectorIndexCommonParams MakeCommon(MetricType metric = MetricType::kL2Distance,
                                   int dim = 64, bool normed = false) {
  VectorIndexCommonParams c;
  c.dim = dim;
  c.metric_type = metric;
  c.is_vector_normed = normed;
  return c;
}

FaissHnswIndexParams MakeParams(ScalarQuantizerType q,
                                int M = 16,
                                int m_pq = 0,
                                int nbits_pq = 8) {
  FaissHnswIndexParams p;
  p.M = M;
  p.quantizer = static_cast<int>(q);
  p.m_pq = m_pq;
  p.nbits_pq = nbits_pq;
  return p;
}

}  // namespace

// ---------- Factory string generation ----------

TEST(HnswFactoryStringTest, FlatDefault) {
  auto common = MakeCommon();
  auto params = MakeParams(ScalarQuantizerType::kFlat, /*M=*/16);
  EXPECT_EQ(faiss_util::GetHnswRepr(common, params), "HNSW16");
}

TEST(HnswFactoryStringTest, FlatWithIDMap) {
  auto common = MakeCommon();
  auto params = MakeParams(ScalarQuantizerType::kFlat, /*M=*/32);
  EXPECT_EQ(faiss_util::GetHnswRepr(common, params, /*use_custom_rowid=*/true),
            "IDMap,HNSW32");
}

TEST(HnswFactoryStringTest, FlatCosineUnnormed) {
  auto common = MakeCommon(MetricType::kCosineSimilarity, /*dim=*/64, /*normed=*/false);
  auto params = MakeParams(ScalarQuantizerType::kFlat, /*M=*/16);
  EXPECT_EQ(faiss_util::GetHnswRepr(common, params), "L2Norm,HNSW16");
}

TEST(HnswFactoryStringTest, Sq4) {
  auto common = MakeCommon();
  auto params = MakeParams(ScalarQuantizerType::kSQ4, /*M=*/16);
  EXPECT_EQ(faiss_util::GetHnswRepr(common, params), "HNSW16,SQ4");
}

TEST(HnswFactoryStringTest, Sq8) {
  auto common = MakeCommon();
  auto params = MakeParams(ScalarQuantizerType::kSQ8, /*M=*/32);
  EXPECT_EQ(faiss_util::GetHnswRepr(common, params), "HNSW32,SQ8");
}

TEST(HnswFactoryStringTest, PqDefaultNbits) {
  auto common = MakeCommon();
  // m_pq=8, nbits_pq=8 (default) -> nbits suffix omitted
  auto params = MakeParams(ScalarQuantizerType::kPQ, /*M=*/16, /*m_pq=*/8, /*nbits_pq=*/8);
  EXPECT_EQ(faiss_util::GetHnswRepr(common, params), "HNSW16,PQ8np");
}

TEST(HnswFactoryStringTest, PqExplicitNbits) {
  auto common = MakeCommon();
  auto params = MakeParams(ScalarQuantizerType::kPQ, /*M=*/16, /*m_pq=*/16, /*nbits_pq=*/6);
  EXPECT_EQ(faiss_util::GetHnswRepr(common, params), "HNSW16,PQ16x6np");
}

// ---------- faiss::index_factory round-trip ----------

TEST(HnswFactoryStringTest, FaissFactoryBuildsHnswFlat) {
  auto common = MakeCommon(MetricType::kL2Distance, /*dim=*/64);
  auto params = MakeParams(ScalarQuantizerType::kFlat, /*M=*/16);
  auto repr = faiss_util::GetHnswRepr(common, params);

  std::unique_ptr<faiss::Index> idx(
      faiss::index_factory(common.dim, repr.c_str(), faiss::METRIC_L2));
  ASSERT_NE(idx, nullptr);
  auto* hnsw = dynamic_cast<faiss::IndexHNSWFlat*>(idx.get());
  EXPECT_NE(hnsw, nullptr);
}

TEST(HnswFactoryStringTest, FaissFactoryBuildsHnswSq8) {
  auto common = MakeCommon(MetricType::kL2Distance, /*dim=*/64);
  auto params = MakeParams(ScalarQuantizerType::kSQ8, /*M=*/16);
  auto repr = faiss_util::GetHnswRepr(common, params);

  std::unique_ptr<faiss::Index> idx(
      faiss::index_factory(common.dim, repr.c_str(), faiss::METRIC_L2));
  ASSERT_NE(idx, nullptr);
  auto* hnsw = dynamic_cast<faiss::IndexHNSWSQ*>(idx.get());
  EXPECT_NE(hnsw, nullptr);
}

TEST(HnswFactoryStringTest, FaissFactoryBuildsHnswSq4) {
  auto common = MakeCommon(MetricType::kL2Distance, /*dim=*/64);
  auto params = MakeParams(ScalarQuantizerType::kSQ4, /*M=*/16);
  auto repr = faiss_util::GetHnswRepr(common, params);

  std::unique_ptr<faiss::Index> idx(
      faiss::index_factory(common.dim, repr.c_str(), faiss::METRIC_L2));
  ASSERT_NE(idx, nullptr);
  auto* hnsw = dynamic_cast<faiss::IndexHNSWSQ*>(idx.get());
  EXPECT_NE(hnsw, nullptr);
}

TEST(HnswFactoryStringTest, FaissFactoryBuildsHnswPq) {
  // dim=64, m_pq=8 => each sub-vector is 8 float dims. nbits=8 -> 256 centroids.
  auto common = MakeCommon(MetricType::kL2Distance, /*dim=*/64);
  auto params = MakeParams(ScalarQuantizerType::kPQ, /*M=*/16, /*m_pq=*/8, /*nbits_pq=*/8);
  auto repr = faiss_util::GetHnswRepr(common, params);
  EXPECT_EQ(repr, "HNSW16,PQ8np");

  std::unique_ptr<faiss::Index> idx(
      faiss::index_factory(common.dim, repr.c_str(), faiss::METRIC_L2));
  ASSERT_NE(idx, nullptr);
  auto* hnsw = dynamic_cast<faiss::IndexHNSWPQ*>(idx.get());
  EXPECT_NE(hnsw, nullptr);
  EXPECT_FALSE(dynamic_cast<faiss::IndexPQ*>(hnsw->storage)->do_polysemous_training);
}

TEST(HnswFactoryStringTest, FaissFactoryBuildsHnswPqCustomNbits) {
  auto common = MakeCommon(MetricType::kL2Distance, /*dim=*/64);
  auto params = MakeParams(ScalarQuantizerType::kPQ, /*M=*/16, /*m_pq=*/8, /*nbits_pq=*/6);
  auto repr = faiss_util::GetHnswRepr(common, params);
  EXPECT_EQ(repr, "HNSW16,PQ8x6np");

  std::unique_ptr<faiss::Index> idx(
      faiss::index_factory(common.dim, repr.c_str(), faiss::METRIC_L2));
  ASSERT_NE(idx, nullptr);
  auto* hnsw = dynamic_cast<faiss::IndexHNSWPQ*>(idx.get());
  EXPECT_NE(hnsw, nullptr);
}

TEST(HnswFactoryStringTest, FaissFactoryBuildsIDMapHnswSq8) {
  auto common = MakeCommon(MetricType::kL2Distance, /*dim=*/64);
  auto params = MakeParams(ScalarQuantizerType::kSQ8, /*M=*/16);
  auto repr = faiss_util::GetHnswRepr(common, params, /*use_custom_rowid=*/true);
  EXPECT_EQ(repr, "IDMap,HNSW16,SQ8");

  std::unique_ptr<faiss::Index> idx(
      faiss::index_factory(common.dim, repr.c_str(), faiss::METRIC_L2));
  ASSERT_NE(idx, nullptr);
  auto* id_map = dynamic_cast<faiss::IndexIDMap*>(idx.get());
  ASSERT_NE(id_map, nullptr);
  EXPECT_NE(dynamic_cast<faiss::IndexHNSWSQ*>(id_map->index), nullptr);
}

// ---------- Logical/physical metric resolution ----------

TEST(VectorMetricResolverTest, ResolvesBuildMetric) {
  EXPECT_EQ(ResolveBuildMetric(MetricType::kL2Distance, CosineBackend::kInnerProduct),
            MetricType::kL2Distance);
  EXPECT_EQ(ResolveBuildMetric(MetricType::kInnerProduct, CosineBackend::kL2),
            MetricType::kInnerProduct);
  EXPECT_EQ(ResolveBuildMetric(MetricType::kCosineSimilarity, CosineBackend::kL2),
            MetricType::kL2Distance);
  EXPECT_EQ(ResolveBuildMetric(MetricType::kCosineSimilarity, CosineBackend::kInnerProduct),
            MetricType::kInnerProduct);
}

TEST(VectorMetricResolverTest, ValidatesLoadedMetric) {
  EXPECT_NO_THROW(ValidateLoadedMetric(MetricType::kL2Distance, MetricType::kL2Distance));
  EXPECT_NO_THROW(ValidateLoadedMetric(MetricType::kInnerProduct, MetricType::kInnerProduct));
  EXPECT_NO_THROW(ValidateLoadedMetric(MetricType::kCosineSimilarity, MetricType::kL2Distance));
  EXPECT_NO_THROW(ValidateLoadedMetric(MetricType::kCosineSimilarity, MetricType::kInnerProduct));
  EXPECT_THROW(ValidateLoadedMetric(MetricType::kL2Distance, MetricType::kInnerProduct), Error);
  EXPECT_THROW(ValidateLoadedMetric(MetricType::kInnerProduct, MetricType::kL2Distance), Error);
}

TEST(VectorMetricResolverTest, RejectsInvalidCosineBackend) {
  EXPECT_EQ(ParseCosineBackend("l2"), CosineBackend::kL2);
  EXPECT_EQ(ParseCosineBackend("inner_product"), CosineBackend::kInnerProduct);
  EXPECT_THROW(ParseCosineBackend("auto"), Error);
}

TEST(HnswPqMemoryTest, EstimatesSdcWithoutProductPolicy) {
  EXPECT_EQ(faiss_util::EstimateHnswPqSdcBytes(96, 4), 96u * 16 * 16 * sizeof(float));
  EXPECT_EQ(faiss_util::EstimateHnswPqSdcBytes(96, 8), 96u * 256 * 256 * sizeof(float));
  EXPECT_EQ(faiss_util::EstimateHnswPqSdcBytes(96, 12), 6442450944ULL);
  EXPECT_EQ(faiss_util::EstimateHnswPqSdcBytes(96, 16), 1649267441664ULL);
  EXPECT_THROW(faiss_util::EstimateHnswPqSdcBytes(
                   2, static_cast<int>(std::numeric_limits<size_t>::digits - 1)),
               Error);
}

// The normalizing pre-transform is emitted for cosine regardless of which metric backs it: the
// inner product form still needs unit-norm vectors for the dot product to BE the cosine.
TEST(HnswCosineMetricTest, QuantizedCosineStillNormalizes) {
  auto common = MakeCommon(MetricType::kCosineSimilarity);
  EXPECT_EQ(faiss_util::GetHnswRepr(common, MakeParams(ScalarQuantizerType::kSQ8)),
            "L2Norm,HNSW16,SQ8");
}

// End to end through faiss: the factory string a quantized cosine index produces, combined with
// METRIC_INNER_PRODUCT, must yield a normalizing pre-transform wrapping an IP-scored HNSW+SQ.
TEST(HnswCosineMetricTest, FaissBuildsNormalizedIpBackedCosineIndex) {
  auto common = MakeCommon(MetricType::kCosineSimilarity, /*dim=*/64);
  auto params = MakeParams(ScalarQuantizerType::kSQ8, /*M=*/16);

  std::unique_ptr<faiss::Index> idx(faiss::index_factory(
      common.dim, faiss_util::GetHnswRepr(common, params).c_str(), faiss::METRIC_INNER_PRODUCT));
  ASSERT_NE(idx, nullptr);

  auto* pre = dynamic_cast<faiss::IndexPreTransform*>(idx.get());
  ASSERT_NE(pre, nullptr) << "cosine must keep its normalizing pre-transform";
  auto* hnsw = dynamic_cast<faiss::IndexHNSWSQ*>(pre->index);
  ASSERT_NE(hnsw, nullptr);
  EXPECT_EQ(hnsw->metric_type, faiss::METRIC_INNER_PRODUCT);
  EXPECT_EQ(hnsw->storage->metric_type, faiss::METRIC_INNER_PRODUCT);
}

TEST(HnswFactoryStringTest, FaissFactoryPropagatesMetricToHnswPq) {
  auto common = MakeCommon(MetricType::kInnerProduct, /*dim=*/64);
  auto params = MakeParams(ScalarQuantizerType::kPQ, /*M=*/16, /*m_pq=*/8, /*nbits_pq=*/4);
  std::unique_ptr<faiss::Index> idx(faiss::index_factory(
      common.dim, faiss_util::GetHnswRepr(common, params).c_str(), faiss::METRIC_INNER_PRODUCT));
  auto* hnsw = dynamic_cast<faiss::IndexHNSWPQ*>(idx.get());
  ASSERT_NE(hnsw, nullptr);
  EXPECT_EQ(hnsw->metric_type, faiss::METRIC_INNER_PRODUCT);
  EXPECT_EQ(hnsw->storage->metric_type, faiss::METRIC_INNER_PRODUCT);
}

}  // namespace tenann
