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

// HNSW Flat built through HnswBuildStorage: rows staged in place, graph built per flush batch below
// the IDMap / L2Norm wrappers, optional 8-bit build distances.

#include <omp.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "faiss/IndexFlat.h"
#include "faiss/IndexHNSW.h"
#include "faiss/IndexIDMap.h"
#include "faiss/IndexPreTransform.h"
#include "faiss/index_factory.h"
#include "faiss/index_io.h"
#include "faiss/utils/distances.h"
#include "gtest/gtest.h"
#include "tenann/common/seq_view.h"
#include "tenann/factory/ann_searcher_factory.h"
#include "tenann/factory/index_factory.h"
#include "tenann/index/internal/faiss_index_util.h"
#include "tenann/index/internal/hnsw_build_storage.h"
#include "tenann/store/index_meta.h"
#include "tenann/store/index_type.h"

namespace tenann {

namespace {

IndexMeta MakeMeta(uint32_t dim, MetricType metric, bool int8_build) {
  IndexMeta meta;
  meta.SetMetaVersion(0);
  meta.SetIndexFamily(IndexFamily::kVectorIndex);
  meta.SetIndexType(IndexType::kFaissHnsw);
  meta.common_params()["dim"] = static_cast<int>(dim);
  meta.common_params()["is_vector_normed"] = false;
  meta.common_params()["metric_type"] = metric;
  meta.index_params()["M"] = 16;
  meta.index_params()["efConstruction"] = 40;
  meta.search_params()["efSearch"] = 64;
  meta.index_writer_options()["write_index_cache"] = false;
  meta.index_writer_options()["hnsw_int8_build"] = int8_build;
  meta.index_reader_options()["cache_index_file"] = false;
  return meta;
}

// Gaussian clusters: a graph over them is a real search problem, unlike uniform noise.
std::vector<float> ClusteredVectors(uint32_t n, uint32_t dim, int seed, float scale = 1.0f) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> g(0.0f, 1.0f);
  const uint32_t clusters = 20;
  std::vector<float> centers(clusters * dim);
  for (auto& c : centers) c = g(rng) * scale;
  std::vector<float> v(static_cast<size_t>(n) * dim);
  for (uint32_t i = 0; i < n; i++) {
    const float* c = centers.data() + (i % clusters) * dim;
    for (uint32_t j = 0; j < dim; j++)
      v[static_cast<size_t>(i) * dim + j] = c[j] + 0.3f * scale * g(rng);
  }
  return v;
}

std::string TempPath(const std::string& tag) {
  std::string t = "/tmp/tenann_hnsw_build_storage_" + tag + "_XXXXXX";
  std::vector<char> buf(t.begin(), t.end());
  buf.push_back('\0');
  int fd = mkstemp(buf.data());
  EXPECT_NE(fd, -1);
  close(fd);
  return buf.data();
}

ArraySeqView View(const std::vector<float>& data, size_t from, size_t rows, uint32_t dim) {
  return ArraySeqView{
      .data = reinterpret_cast<uint8_t*>(const_cast<float*>(data.data() + from * dim)),
      .dim = dim,
      .size = static_cast<uint32_t>(rows),
      .elem_type = PrimitiveType::kFloatType};
}

struct CountingAccountant : MemoryAccountant {
  void Charge(int64_t bytes) override {
    int64_t now = current += bytes;
    int64_t p = peak.load();
    while (now > p && !peak.compare_exchange_weak(p, now)) {
    }
  }
  void Release(int64_t bytes) override { current -= bytes; }
  std::atomic<int64_t> current{0};
  std::atomic<int64_t> peak{0};
};

// Feeds |data| the way StarRocks does: fixed-size batches with row ids and per-row null flags.
void Build(const IndexMeta& meta, const std::vector<float>& data, uint32_t dim, size_t flush_rows,
           const std::string& path, const std::vector<uint8_t>* nulls = nullptr,
           MemoryAccountantPtr accountant = nullptr) {
  const size_t n = data.size() / dim;
  const size_t batch = 1000;
  auto builder = IndexFactory::CreateBuilderFromMeta(meta);
  builder->SetFlushThresholdRows(flush_rows);
  if (accountant) builder->SetMemoryAccountant(accountant);
  builder->EnableCustomRowId();
  builder->Open(path);
  std::vector<int64_t> ids(batch);
  std::vector<uint8_t> no_nulls(batch, 0);
  for (size_t off = 0; off < n; off += batch) {
    size_t m = std::min(batch, n - off);
    std::iota(ids.begin(), ids.begin() + m, static_cast<int64_t>(off));
    const uint8_t* nf = nulls ? nulls->data() + off : no_nulls.data();
    builder->Add({View(data, off, m, dim)}, ids.data(), nf);
  }
  builder->Flush();
  builder->Close();
}

// Exact top-k row ids over the (optionally normalized) rows that are not null.
std::vector<int64_t> ExactTopK(std::vector<float> base, const std::vector<float>& queries,
                               uint32_t dim, size_t k, bool normalize,
                               const std::vector<uint8_t>* nulls = nullptr) {
  const size_t n = base.size() / dim, nq = queries.size() / dim;
  std::vector<float> q = queries;
  if (normalize) {
    faiss::fvec_renorm_L2(dim, n, base.data());
    faiss::fvec_renorm_L2(dim, nq, q.data());
  }
  faiss::IndexFlatL2 flat(dim);
  faiss::IndexIDMap idmap(&flat);
  std::vector<int64_t> ids;
  std::vector<float> rows;
  for (size_t i = 0; i < n; i++) {
    if (nulls && (*nulls)[i]) continue;
    ids.push_back(i);
    rows.insert(rows.end(), base.begin() + i * dim, base.begin() + (i + 1) * dim);
  }
  idmap.add_with_ids(ids.size(), rows.data(), ids.data());
  std::vector<float> dist(nq * k);
  std::vector<int64_t> out(nq * k);
  idmap.search(nq, q.data(), k, dist.data(), out.data());
  return out;
}

// Recall@k of the written index, searched through the searcher StarRocks uses.
float Recall(const IndexMeta& meta, const std::string& path, const std::vector<float>& queries,
             uint32_t dim, size_t k, const std::vector<int64_t>& truth, int ef_search = 64) {
  auto searcher = AnnSearcherFactory::CreateSearcherFromMeta(meta);
  searcher->ReadIndex(path);
  searcher->SetSearchParamItem("efSearch", ef_search);
  const size_t nq = queries.size() / dim;
  std::vector<int64_t> got(k);
  size_t hits = 0;
  for (size_t i = 0; i < nq; i++) {
    auto qv = PrimitiveSeqView{
        .data = reinterpret_cast<uint8_t*>(const_cast<float*>(queries.data() + i * dim)),
        .size = dim,
        .elem_type = PrimitiveType::kFloatType};
    searcher->AnnSearch(qv, k, got.data());
    for (size_t a = 0; a < k; a++)
      for (size_t b = 0; b < k; b++)
        if (got[a] == truth[i * k + b]) {
          hits++;
          break;
        }
  }
  return static_cast<float>(hits) / (nq * k);
}

struct OmpThreads {
  explicit OmpThreads(int n) : saved(omp_get_max_threads()) { omp_set_num_threads(n); }
  ~OmpThreads() { omp_set_num_threads(saved); }
  int saved;
};

}  // namespace

// With one thread, staging rows in place and building below the wrappers must give the graph
// faiss's own add_with_ids builds for the same batches: same neighbors, levels, row ids and
// (normalized) rows, whether the rows are on the heap or in a mapping charged to an accountant.
// 6000 rows also outgrow the storage's first 4096 rows while rows are staged.
TEST(HnswBuildStorageTest, SameIndexAsFaissAdd) {
  OmpThreads one(1);
  const uint32_t dim = 32;
  const size_t n = 6000, flush = 2500;
  auto data = ClusteredVectors(n, dim, 7);
  for (bool mapped : {false, true})
    for (MetricType metric : {MetricType::kCosineSimilarity, MetricType::kL2Distance}) {
      SCOPED_TRACE(std::to_string(static_cast<int>(metric)) + (mapped ? " mapped" : " heap"));
      auto meta = MakeMeta(dim, metric, /*int8_build=*/false);
      auto path = TempPath("same");
      Build(meta, data, dim, flush, path, nullptr,
            mapped ? std::make_shared<CountingAccountant>() : nullptr);
      std::unique_ptr<faiss::Index> got(faiss::read_index(path.c_str()));

      // Reference: the factory string tenann uses, rows added in the same batches.
      VectorIndexCommonParams cp;
      cp.dim = dim;
      cp.metric_type = static_cast<int>(metric);
      cp.is_vector_normed = false;
      FaissHnswIndexParams hp;
      hp.M = 16;
      auto repr = faiss_util::GetHnswRepr(cp, hp, /*use_custom_rowid=*/true);
      std::unique_ptr<faiss::Index> ref(faiss::index_factory(dim, repr.c_str(), faiss::METRIC_L2));
      auto [ref_ids, ref_tr, ref_hnsw] = faiss_util::UnpackHnswMutable(ref.get());
      ref_hnsw->hnsw.efConstruction = 40;
      std::vector<int64_t> ids(n);
      std::iota(ids.begin(), ids.end(), 0);
      for (size_t off = 0; off < n; off += flush) {
        size_t m = std::min(flush, n - off);
        ref->add_with_ids(m, data.data() + off * dim, ids.data() + off);
      }

      auto [got_ids, got_tr, got_hnsw] = faiss_util::UnpackHnswMutable(got.get());
      ASSERT_EQ(got->ntotal, static_cast<faiss::idx_t>(n));
      ASSERT_EQ((got_tr != nullptr), (ref_tr != nullptr));
      if (got_tr != nullptr) EXPECT_EQ(got_tr->ntotal, static_cast<faiss::idx_t>(n));
      EXPECT_EQ(got_ids->id_map, ref_ids->id_map);
      EXPECT_EQ(got_hnsw->hnsw.levels, ref_hnsw->hnsw.levels);
      EXPECT_EQ(got_hnsw->hnsw.offsets, ref_hnsw->hnsw.offsets);
      const auto& gn = got_hnsw->hnsw.neighbors;
      const auto& rn = ref_hnsw->hnsw.neighbors;
      ASSERT_EQ(gn.size(), rn.size());
      EXPECT_TRUE(std::equal(gn.data(), gn.data() + gn.size(), rn.data()));
      EXPECT_EQ(got_hnsw->hnsw.entry_point, ref_hnsw->hnsw.entry_point);
      auto* got_flat = dynamic_cast<faiss::IndexFlatL2*>(got_hnsw->storage);
      auto* ref_flat = dynamic_cast<faiss::IndexFlat*>(ref_hnsw->storage);
      ASSERT_NE(got_flat, nullptr) << "the build storage must be written as a plain IndexFlatL2";
      ASSERT_EQ(got_flat->codes.size(), ref_flat->codes.size());
      EXPECT_EQ(0, memcmp(got_flat->codes.data(), ref_flat->codes.data(), ref_flat->codes.size()));
      std::remove(path.c_str());
    }
}

// Every mapped byte is charged while held: the rows for the index's lifetime, the 8-bit codes until
// Flush(). The charge stays close to what is actually stored.
TEST(HnswBuildStorageTest, AccountantChargedForMappedRows) {
  const uint32_t dim = 32;
  const size_t n = 50000;
  auto data = ClusteredVectors(n, dim, 41);
  auto accountant = std::make_shared<CountingAccountant>();
  auto meta = MakeMeta(dim, MetricType::kCosineSimilarity, /*int8_build=*/true);
  auto path = TempPath("accountant");
  {
    auto builder = IndexFactory::CreateBuilderFromMeta(meta);
    builder->SetFlushThresholdRows(20000);
    builder->SetMemoryAccountant(accountant);
    builder->EnableCustomRowId();
    builder->Open(path);
    std::vector<int64_t> ids(n);
    std::iota(ids.begin(), ids.end(), 0);
    std::vector<uint8_t> nulls(n, 0);
    builder->Add({View(data, 0, n, dim)}, ids.data(), nulls.data());
    const int64_t rows_bytes = static_cast<int64_t>(n) * dim * sizeof(float);
    const int64_t codes_bytes = static_cast<int64_t>(n) * dim;
    builder->Flush();
    EXPECT_GE(accountant->current.load(), rows_bytes);
    EXPECT_LE(accountant->current.load(), rows_bytes * 1.15 + (1 << 20));  // codes released
    EXPECT_GE(accountant->peak.load(), rows_bytes + codes_bytes);
    EXPECT_LE(accountant->peak.load(), (rows_bytes + codes_bytes) * 1.15 + (2 << 20));
    builder->Close();
  }
  EXPECT_EQ(accountant->current.load(), 0);
  std::remove(path.c_str());
}

TEST(HnswBuildStorageTest, NullRowsSkippedAndRowIdsKept) {
  const uint32_t dim = 32;
  const size_t n = 3000, k = 10;
  auto data = ClusteredVectors(n, dim, 21);
  std::vector<uint8_t> nulls(n, 0);
  for (size_t i = 0; i < n; i += 7) nulls[i] = 1;
  auto meta = MakeMeta(dim, MetricType::kL2Distance, true);
  auto path = TempPath("nulls");
  Build(meta, data, dim, 1000, path, &nulls);
  std::unique_ptr<faiss::Index> idx(faiss::read_index(path.c_str()));
  EXPECT_EQ(idx->ntotal, static_cast<faiss::idx_t>(n - (n + 6) / 7));
  // Querying with the rows themselves finds their own row ids and never a null row.
  std::vector<float> self(data.begin(), data.begin() + 200 * dim);
  auto truth = ExactTopK(data, self, dim, k, false, &nulls);
  EXPECT_GE(Recall(meta, path, self, dim, k, truth), 0.95f);
  std::remove(path.c_str());
}

// Rows on a low-dimensional manifold, like real embeddings: an 8-d latent point through a fixed
// random projection (|proj_seed|), plus a little noise. Isotropic high-dimensional noise has no
// near neighbors to find.
std::vector<float> ManifoldVectors(uint32_t n, uint32_t dim, int proj_seed, int seed) {
  const uint32_t latent = 8;
  std::mt19937 prng(proj_seed), rng(seed);
  std::normal_distribution<float> g(0.0f, 1.0f);
  std::vector<float> w(latent * dim);
  for (auto& x : w) x = g(prng);
  std::vector<float> v(static_cast<size_t>(n) * dim);
  std::vector<float> z(latent);
  for (uint32_t i = 0; i < n; i++) {
    for (auto& x : z) x = g(rng);
    for (uint32_t j = 0; j < dim; j++) {
      float s = 0.05f * g(rng);
      for (uint32_t l = 0; l < latent; l++) s += z[l] * w[l * dim + j];
      v[static_cast<size_t>(i) * dim + j] = s;
    }
  }
  return v;
}

// 8-bit build distances only steer which edges are kept; the written index is float and searched as
// before.
TEST(HnswBuildStorageTest, Int8BuildRecallMatchesFloatBuild) {
  struct Case {
    MetricType metric;
    uint32_t dim;  // 37: exercises the kernels' scalar tail
  };
  for (Case c : {Case{MetricType::kCosineSimilarity, 64}, Case{MetricType::kL2Distance, 37}}) {
    SCOPED_TRACE(static_cast<int>(c.metric));
    const size_t n = 8000, k = 10;
    auto data = ManifoldVectors(n, c.dim, 5, 11);
    auto queries = ManifoldVectors(200, c.dim, 5, 12);
    const bool cosine = c.metric == MetricType::kCosineSimilarity;
    auto truth = ExactTopK(data, queries, c.dim, k, cosine);
    float recall[2];
    for (int int8 = 0; int8 < 2; int8++) {
      auto meta = MakeMeta(c.dim, c.metric, int8 == 1);
      auto path = TempPath("recall");
      Build(meta, data, c.dim, 3000, path);
      // A small ef keeps recall below 1, where a worse graph shows.
      recall[int8] = Recall(meta, path, queries, c.dim, k, truth, /*ef_search=*/10);
      std::remove(path.c_str());
    }
    EXPECT_GE(recall[0], 0.8f) << "float build";
    EXPECT_GE(recall[1], recall[0] - 0.02f) << "float " << recall[0] << " int8 " << recall[1];
    printf("metric %d dim %u: recall@10 float build %.4f, int8 build %.4f\n",
           static_cast<int>(c.metric), c.dim, recall[0], recall[1]);
  }
}

// L2 rows have no known range: the 8-bit range comes from the first batch and must widen
// (re-encoding what is already in the graph) when a later batch falls outside it. Clipping the
// later rows instead collapses their codes, and the graph over them falls behind one built with
// float distances.
TEST(HnswBuildStorageTest, Int8RangeWidensForLaterBatches) {
  const uint32_t dim = 32;
  const size_t n = 6000, k = 10;
  auto data = ManifoldVectors(n, dim, 5, 31);
  for (size_t i = (n / 2) * dim; i < n * dim; i++)
    data[i] *= 20.0f;  // rows the first batch never saw
  std::vector<float> queries = ManifoldVectors(200, dim, 5, 32);
  for (auto& x : queries) x *= 20.0f;
  auto truth = ExactTopK(data, queries, dim, k, false);
  float recall[2];
  for (int int8 = 0; int8 < 2; int8++) {
    auto meta = MakeMeta(dim, MetricType::kL2Distance, int8 == 1);
    auto path = TempPath("range");
    Build(meta, data, dim, 2000, path);
    recall[int8] = Recall(meta, path, queries, dim, k, truth, /*ef_search=*/10);
    std::remove(path.c_str());
  }
  // One range over both scales coarsens the small rows a little (about 1.5 points here); clipping
  // costs ~10.
  EXPECT_GE(recall[1], recall[0] - 0.05f) << "float " << recall[0] << " int8 " << recall[1];
  printf("widened range: recall@10 float build %.4f, int8 build %.4f\n", recall[0], recall[1]);
}

// Every SIMD kernel this CPU has must give the scalar result exactly: tails of every width, codes
// at 0 and 255, and dimensions past the 8192-dim point where the int32 lanes are folded.
TEST(HnswBuildStorageTest, Int8KernelsMatchScalar) {
  std::mt19937 rng(3);
  std::uniform_int_distribution<int> byte(0, 255);
  for (size_t d : {1, 15, 16, 17, 31, 32, 33, 37, 64, 768, 1000, 16391}) {
    std::vector<std::vector<uint8_t>> rows(5, std::vector<uint8_t>(d));
    for (auto& r : rows)
      for (auto& x : r) x = static_cast<uint8_t>(byte(rng));
    std::fill(rows[1].begin(), rows[1].end(),
              255);  // largest per-dimension term against rows[0]...
    std::fill(rows[0].begin(), rows[0].begin() + d / 2, 0);
    const uint8_t* y[4] = {rows[1].data(), rows[2].data(), rows[3].data(), rows[4].data()};
    int64_t want_one, want_four[4];
    ASSERT_TRUE(
        hnsw_build_internal::L2U8WithLevel("scalar", rows[0].data(), y, d, &want_one, want_four));
    for (const char* level : {"avx2", "avx512"}) {
      int64_t one, four[4];
      if (!hnsw_build_internal::L2U8WithLevel(level, rows[0].data(), y, d, &one, four)) continue;
      SCOPED_TRACE(std::string(level) + " d=" + std::to_string(d));
      EXPECT_EQ(one, want_one);
      for (int k = 0; k < 4; k++) EXPECT_EQ(four[k], want_four[k]);
    }
  }
}

}  // namespace tenann
