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

#include "tenann/index/internal/index_ivfpq.h"

#include <omp.h>
#include <stdint.h>

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <mutex>

#include "faiss/Clustering.h"
#include "faiss/IndexFlat.h"
#include "faiss/impl/AuxIndexStructures.h"
#include "faiss/impl/FaissAssert.h"
#include "faiss/impl/IDSelector.h"
#include "faiss/impl/ProductQuantizer.h"
#include "faiss/utils/Heap.h"
#include "faiss/utils/distances.h"
#include "faiss/utils/hamming.h"
#include "faiss/utils/utils.h"

#ifdef __AVX2__
#include <immintrin.h>

#include "index_ivfpq.h"
#endif

namespace tenann {
using namespace faiss;

IndexIvfPq::IndexIvfPq() : faiss::IndexIVFPQ() {}

IndexIvfPq::~IndexIvfPq() {}

static float* compute_residuals(const Index* quantizer, Index::idx_t n, const float* x,
                                const Index::idx_t* list_nos) {
  size_t d = quantizer->d;
  float* residuals = new float[n * d];
  // TODO: parallelize?
  for (size_t i = 0; i < n; i++) {
    if (list_nos[i] < 0)
      memset(residuals + i * d, 0, sizeof(*residuals) * d);
    else
      quantizer->compute_residual(x + i * d, residuals + i * d, list_nos[i]);
  }
  return residuals;
}

IndexIvfPq::IndexIvfPq(faiss::Index* quantizer, size_t d, size_t nlist, size_t M,
                       size_t nbits_per_idx)
    : IndexIVFPQ(quantizer, d, nlist, M, nbits_per_idx, faiss::METRIC_L2) {
  /* The following lines are added by tenann */
  reconstruction_errors.resize(nlist);
  /* End tenann.*/
}

void IndexIvfPq::add_core(idx_t n, const float* x, const idx_t* xids, const idx_t* coarse_idx) {
  // add_core_o(n, x, xids, nullptr, coarse_idx);

  /* The following lines are added by tenann */
  custom_add_core_o(n, x, xids, nullptr, coarse_idx);
  /* End tenann.*/
}

// block size used in IndexIvfPq::add_core_o
static int index_ivfpq_add_core_o_bs = 32768;

void IndexIvfPq::custom_add_core_o(idx_t n, const float* x, const idx_t* xids, float* residuals_2,
                                   const idx_t* precomputed_idx) {
  idx_t bs = index_ivfpq_add_core_o_bs;
  if (n > bs) {
    for (idx_t i0 = 0; i0 < n; i0 += bs) {
      idx_t i1 = std::min(i0 + bs, n);
      if (verbose) {
        printf("IndexIvfPq::add_core_o: adding %" PRId64 ":%" PRId64 " / %" PRId64 "\n", i0, i1, n);
      }
      custom_add_core_o(i1 - i0, x + i0 * d, xids ? xids + i0 : nullptr,
                        residuals_2 ? residuals_2 + i0 * d : nullptr,
                        precomputed_idx ? precomputed_idx + i0 : nullptr);
    }
    return;
  }

  InterruptCallback::check();

  direct_map.check_can_add(xids);

  FAISS_THROW_IF_NOT(is_trained);
  double t0 = getmillisecs();
  const idx_t* idx;
  ScopeDeleter<idx_t> del_idx;

  if (precomputed_idx) {
    idx = precomputed_idx;
  } else {
    idx_t* idx0 = new idx_t[n];
    del_idx.set(idx0);
    quantizer->assign(n, x, idx0);
    idx = idx0;
  }

  double t1 = getmillisecs();
  uint8_t* xcodes = new uint8_t[n * code_size];
  ScopeDeleter<uint8_t> del_xcodes(xcodes);

  const float* to_encode = nullptr;
  ScopeDeleter<float> del_to_encode;

  if (by_residual) {
    to_encode = compute_residuals(quantizer, n, x, idx);
    del_to_encode.set(to_encode);
  } else {
    to_encode = x;
  }
  pq.compute_codes(to_encode, xcodes, n);

  double t2 = getmillisecs();
  // TODO: parallelize?
  size_t n_ignore = 0;
  for (size_t i = 0; i < n; i++) {
    idx_t key = idx[i];
    idx_t id = xids ? xids[i] : ntotal + i;
    if (key < 0) {
      direct_map.add_single_id(id, -1, 0);
      n_ignore++;
      if (residuals_2) memset(residuals_2, 0, sizeof(*residuals_2) * d);
      continue;
    }

    uint8_t* code = xcodes + i * code_size;
    size_t offset = invlists->add_entry(key, id, code);

    /* The following lines are modified by tenann */
    std::vector<float> vector_res2(d);
    float* res2 = residuals_2 ? residuals_2 + i * d : vector_res2.data();

    const float* xi = to_encode + i * d;
    pq.decode(code, res2);
    for (int j = 0; j < d; j++) res2[j] = xi[j] - res2[j];

    float reconstruction_error;
    fvec_norms_L2(&reconstruction_error, res2, d, 1);
    reconstruction_errors[key].push_back(reconstruction_error);
    FAISS_THROW_IF_NOT(reconstruction_errors[key].size() != offset);
    /* End tenann.*/

    direct_map.add_single_id(id, key, offset);
  }

  double t3 = getmillisecs();
  if (verbose) {
    char comment[100] = {0};
    if (n_ignore > 0) snprintf(comment, 100, "(%zd vectors ignored)", n_ignore);
    printf(" add_core times: %.3f %.3f %.3f %s\n", t1 - t0, t2 - t1, t3 - t2, comment);
  }
  ntotal += n;
}

// Ported from faiss/IndexIVFPQ.cpp
void IndexIvfPq::range_search(idx_t nx, const float* x, float radius, RangeSearchResult* result,
                              const SearchParameters* params_in) const {
  const IndexIvfPqSearchParameters* params = nullptr;
  const SearchParameters* quantizer_params = nullptr;
  if (params_in) {
    params = dynamic_cast<const IndexIvfPqSearchParameters*>(params_in);
    FAISS_THROW_IF_NOT_MSG(params, "IndexIvfPq params have incorrect type");
    quantizer_params = params->quantizer_params;
  }
  const size_t nprobe = std::min(nlist, params ? params->nprobe : this->nprobe);
  std::unique_ptr<idx_t[]> keys(new idx_t[nx * nprobe]);
  std::unique_ptr<float[]> coarse_dis(new float[nx * nprobe]);

  double t0 = getmillisecs();
  quantizer->search(nx, x, nprobe, coarse_dis.get(), keys.get(), quantizer_params);
  indexIVF_stats.quantization_time += getmillisecs() - t0;

  t0 = getmillisecs();
  invlists->prefetch_lists(keys.get(), nx * nprobe);

  custom_range_search_preassigned(nx, x, radius, keys.get(), coarse_dis.get(), result, false,
                                  params, &indexIVF_stats);

  indexIVF_stats.search_time += getmillisecs() - t0;
}

void IndexIvfPq::custom_range_search_preassigned(idx_t nx, const float* x, float radius,
                                                 const idx_t* keys, const float* coarse_dis,
                                                 RangeSearchResult* result, bool store_pairs,
                                                 const IndexIvfPqSearchParameters* params,
                                                 IndexIVFStats* stats) const {
  idx_t nprobe = params ? params->nprobe : this->nprobe;

  /* The following lines are added by tenann */
  float dynamic_range_search_confidence =
      params ? params->range_search_confidence : this->range_search_confidence;
  /* End tenann.*/

  nprobe = std::min((idx_t)nlist, nprobe);
  FAISS_THROW_IF_NOT(nprobe > 0);

  idx_t max_codes = params ? params->max_codes : this->max_codes;
  IDSelector* sel = params ? params->sel : nullptr;

  size_t nlistv = 0, ndis = 0;

  bool interrupt = false;
  std::mutex exception_mutex;
  std::string exception_string;

  std::vector<RangeSearchPartialResult*> all_pres(omp_get_max_threads());

  int pmode = this->parallel_mode & ~PARALLEL_MODE_NO_HEAP_INIT;
  // don't start parallel section if single query
  bool do_parallel = omp_get_max_threads() >= 2 && (pmode == 3   ? false
                                                    : pmode == 0 ? nx > 1
                                                    : pmode == 1 ? nprobe > 1
                                                                 : nprobe * nx > 1);

#pragma omp parallel if (do_parallel) reduction(+ : nlistv, ndis)
  {
    RangeSearchPartialResult pres(result);
    std::unique_ptr<InvertedListScanner> scanner(custom_get_InvertedListScanner(
        store_pairs, sel, dynamic_range_search_confidence));  // modifid by tenann
    FAISS_THROW_IF_NOT(scanner.get());
    all_pres[omp_get_thread_num()] = &pres;

    // prepare the list scanning function

    auto scan_list_func = [&](size_t i, size_t ik, RangeQueryResult& qres) {
      idx_t key = keys[i * nprobe + ik]; /* select the list  */
      if (key < 0) return;
      FAISS_THROW_IF_NOT_FMT(key < (idx_t)nlist, "Invalid key=%" PRId64 " at ik=%zd nlist=%zd\n",
                             key, ik, nlist);
      const size_t list_size = invlists->list_size(key);

      if (list_size == 0) return;

      try {
        InvertedLists::ScopedCodes scodes(invlists, key);
        InvertedLists::ScopedIds ids(invlists, key);

        scanner->set_list(key, coarse_dis[i * nprobe + ik]);
        nlistv++;
        ndis += list_size;
        scanner->scan_codes_range(list_size, scodes.get(), ids.get(), radius, qres);

      } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lock(exception_mutex);
        exception_string = demangle_cpp_symbol(typeid(e).name()) + "  " + e.what();
        interrupt = true;
      }
    };

    if (parallel_mode == 0) {
#pragma omp for
      for (idx_t i = 0; i < nx; i++) {
        scanner->set_query(x + i * d);

        RangeQueryResult& qres = pres.new_result(i);

        for (size_t ik = 0; ik < nprobe; ik++) {
          scan_list_func(i, ik, qres);
        }
      }

    } else if (parallel_mode == 1) {
      for (size_t i = 0; i < nx; i++) {
        scanner->set_query(x + i * d);

        RangeQueryResult& qres = pres.new_result(i);

#pragma omp for schedule(dynamic)
        for (int64_t ik = 0; ik < nprobe; ik++) {
          scan_list_func(i, ik, qres);
        }
      }
    } else if (parallel_mode == 2) {
      RangeQueryResult* qres = nullptr;

#pragma omp for schedule(dynamic)
      for (idx_t iik = 0; iik < nx * (idx_t)nprobe; iik++) {
        idx_t i = iik / (idx_t)nprobe;
        idx_t ik = iik % (idx_t)nprobe;
        if (qres == nullptr || qres->qno != i) {
          qres = &pres.new_result(i);
          scanner->set_query(x + i * d);
        }
        scan_list_func(i, ik, *qres);
      }
    } else {
      FAISS_THROW_FMT("parallel_mode %d not supported\n", parallel_mode);
    }
    if (parallel_mode == 0) {
      pres.finalize();
    } else {
#pragma omp barrier
#pragma omp single
      RangeSearchPartialResult::merge(all_pres, false);
#pragma omp barrier
    }
  }

  if (interrupt) {
    if (!exception_string.empty()) {
      FAISS_THROW_FMT("search interrupted with: %s", exception_string.c_str());
    } else {
      FAISS_THROW_MSG("computation interrupted");
    }
  }

  if (stats) {
    stats->nq += nx;
    stats->nlist += nlistv;
    stats->ndis += ndis;
  }
}

/// 2G by default, accommodates tables up to PQ32 w/ 65536 centroids
static size_t precomputed_table_max_bytes = ((size_t)1) << 31;

namespace {

using idx_t = Index::idx_t;

#define TIC t0 = get_cycles()
#define TOC get_cycles() - t0

/** QueryTables manages the various ways of searching an
 * IndexIvfPq. The code contains a lot of branches, depending on:
 * - metric_type: are we computing L2 or Inner product similarity?
 * - by_residual: do we encode raw vectors or residuals?
 * - use_precomputed_table: are x_R|x_C tables precomputed?
 * - polysemous_ht: are we filtering with polysemous codes?
 */
struct QueryTables {
  /*****************************************************
   * General data from the IVFPQ
   *****************************************************/

  const IndexIvfPq& ivfpq;
  const IVFSearchParameters* params;

  // copied from IndexIvfPq for easier access
  int d;
  const ProductQuantizer& pq;
  MetricType metric_type;
  bool by_residual;
  int use_precomputed_table;
  int polysemous_ht;

  /* The following lines are added by tenann */
  /*
   *  1. by_residual ==  true:
   *    d = || x - y_C ||^2 + || y_R ||^2 + 2 * (y_C|y_R) - 2 * (x|y_R)
   *        ---------------   ---------------------------       -------
   *             term 1                 term 2                   term 3
   *
   *    sim_table = term1 + term2 + term3, sim_table_2 = term3
   *  2. by_residual == false:
   *    d = || x - y_R ||^2
   *        ---------------
   *             term
   *
   *    sim_table = term, sim_table_2 = nullptr
   */
  /* End tenann. */
  // pre-allocated data buffers
  float *sim_table, *sim_table_2;
  float *residual_vec, *decoded_vec;

  // single data buffer
  std::vector<float> mem;

  // for table pointers
  std::vector<const float*> sim_table_ptrs;

  explicit QueryTables(const IndexIvfPq& ivfpq, const IVFSearchParameters* params)
      : ivfpq(ivfpq),
        d(ivfpq.d),
        pq(ivfpq.pq),
        metric_type(ivfpq.metric_type),
        by_residual(ivfpq.by_residual),
        use_precomputed_table(ivfpq.use_precomputed_table) {
    mem.resize(pq.ksub * pq.M * 2 + d * 2);
    sim_table = mem.data();
    sim_table_2 = sim_table + pq.ksub * pq.M;
    residual_vec = sim_table_2 + pq.ksub * pq.M;
    decoded_vec = residual_vec + d;

    // for polysemous
    polysemous_ht = ivfpq.polysemous_ht;
    if (auto ivfpq_params = dynamic_cast<const IVFPQSearchParameters*>(params)) {
      polysemous_ht = ivfpq_params->polysemous_ht;
    }
    if (polysemous_ht != 0) {
      q_code.resize(pq.code_size);
    }
    init_list_cycles = 0;
    sim_table_ptrs.resize(pq.M);
  }

  /*****************************************************
   * What we do when query is known
   *****************************************************/

  // field specific to query
  const float* qi;

  // query-specific initialization
  void init_query(const float* qi) {
    this->qi = qi;
    if (metric_type == METRIC_INNER_PRODUCT)
      init_query_IP();
    else
      init_query_L2();
    if (!by_residual && polysemous_ht != 0) pq.compute_code(qi, q_code.data());
  }

  void init_query_IP() {
    // precompute some tables specific to the query qi
    pq.compute_inner_prod_table(qi, sim_table);
  }

  void init_query_L2() {
    if (!by_residual) {
      pq.compute_distance_table(qi, sim_table);
    } else if (use_precomputed_table) {
      pq.compute_inner_prod_table(qi, sim_table_2);
    }
  }

  /*****************************************************
   * When inverted list is known: prepare computations
   *****************************************************/

  // fields specific to list
  Index::idx_t key;
  float coarse_dis;
  std::vector<uint8_t> q_code;

  uint64_t init_list_cycles;

  /// once we know the query and the centroid, we can prepare the
  /// sim_table that will be used for accumulation
  /// and dis0, the initial value
  float precompute_list_tables() {
    float dis0 = 0;
    uint64_t t0;
    TIC;
    if (by_residual) {
      if (metric_type == METRIC_INNER_PRODUCT)
        dis0 = precompute_list_tables_IP();
      else
        dis0 = precompute_list_tables_L2();
    }
    init_list_cycles += TOC;
    return dis0;
  }

  float precompute_list_table_pointers() {
    float dis0 = 0;
    uint64_t t0;
    TIC;
    if (by_residual) {
      if (metric_type == METRIC_INNER_PRODUCT)
        FAISS_THROW_MSG("not implemented");
      else
        dis0 = precompute_list_table_pointers_L2();
    }
    init_list_cycles += TOC;
    return dis0;
  }

  /*****************************************************
   * compute tables for inner prod
   *****************************************************/

  float precompute_list_tables_IP() {
    // prepare the sim_table that will be used for accumulation
    // and dis0, the initial value
    ivfpq.quantizer->reconstruct(key, decoded_vec);
    // decoded_vec = centroid
    float dis0 = fvec_inner_product(qi, decoded_vec, d);

    if (polysemous_ht) {
      for (int i = 0; i < d; i++) {
        residual_vec[i] = qi[i] - decoded_vec[i];
      }
      pq.compute_code(residual_vec, q_code.data());
    }
    return dis0;
  }

  /*****************************************************
   * compute tables for L2 distance
   *****************************************************/

  float precompute_list_tables_L2() {
    float dis0 = 0;

    if (use_precomputed_table == 0 || use_precomputed_table == -1) {
      ivfpq.quantizer->compute_residual(qi, residual_vec, key);
      pq.compute_distance_table(residual_vec, sim_table);

      if (polysemous_ht != 0) {
        pq.compute_code(residual_vec, q_code.data());
      }

    } else if (use_precomputed_table == 1) {
      dis0 = coarse_dis;

      fvec_madd(pq.M * pq.ksub, ivfpq.precomputed_table.data() + key * pq.ksub * pq.M, -2.0,
                sim_table_2, sim_table);

      if (polysemous_ht != 0) {
        ivfpq.quantizer->compute_residual(qi, residual_vec, key);
        pq.compute_code(residual_vec, q_code.data());
      }

    } else if (use_precomputed_table == 2) {
      dis0 = coarse_dis;

      const MultiIndexQuantizer* miq = dynamic_cast<const MultiIndexQuantizer*>(ivfpq.quantizer);
      FAISS_THROW_IF_NOT(miq);
      const ProductQuantizer& cpq = miq->pq;
      int Mf = pq.M / cpq.M;

      const float* qtab = sim_table_2;  // query-specific table
      float* ltab = sim_table;          // (output) list-specific table

      long k = key;
      for (int cm = 0; cm < cpq.M; cm++) {
        // compute PQ index
        int ki = k & ((uint64_t(1) << cpq.nbits) - 1);
        k >>= cpq.nbits;

        // get corresponding table
        const float* pc = ivfpq.precomputed_table.data() + (ki * pq.M + cm * Mf) * pq.ksub;

        if (polysemous_ht == 0) {
          // sum up with query-specific table
          fvec_madd(Mf * pq.ksub, pc, -2.0, qtab, ltab);
          ltab += Mf * pq.ksub;
          qtab += Mf * pq.ksub;
        } else {
          for (int m = cm * Mf; m < (cm + 1) * Mf; m++) {
            q_code[m] = fvec_madd_and_argmin(pq.ksub, pc, -2, qtab, ltab);
            pc += pq.ksub;
            ltab += pq.ksub;
            qtab += pq.ksub;
          }
        }
      }
    }

    return dis0;
  }

  float precompute_list_table_pointers_L2() {
    float dis0 = 0;

    if (use_precomputed_table == 1) {
      dis0 = coarse_dis;

      const float* s = ivfpq.precomputed_table.data() + key * pq.ksub * pq.M;
      for (int m = 0; m < pq.M; m++) {
        sim_table_ptrs[m] = s;
        s += pq.ksub;
      }
    } else if (use_precomputed_table == 2) {
      dis0 = coarse_dis;

      const MultiIndexQuantizer* miq = dynamic_cast<const MultiIndexQuantizer*>(ivfpq.quantizer);
      FAISS_THROW_IF_NOT(miq);
      const ProductQuantizer& cpq = miq->pq;
      int Mf = pq.M / cpq.M;

      long k = key;
      int m0 = 0;
      for (int cm = 0; cm < cpq.M; cm++) {
        int ki = k & ((uint64_t(1) << cpq.nbits) - 1);
        k >>= cpq.nbits;

        const float* pc = ivfpq.precomputed_table.data() + (ki * pq.M + cm * Mf) * pq.ksub;

        for (int m = m0; m < m0 + Mf; m++) {
          sim_table_ptrs[m] = pc;
          pc += pq.ksub;
        }
        m0 += Mf;
      }
    } else {
      FAISS_THROW_MSG("need precomputed tables");
    }

    if (polysemous_ht) {
      FAISS_THROW_MSG("not implemented");
      // Not clear that it makes sense to implemente this,
      // because it costs M * ksub, which is what we wanted to
      // avoid with the tables pointers.
    }

    return dis0;
  }
};

template <class C, bool use_sel, bool use_range_search_confidence>
struct RangeSearchResults {
  idx_t key;
  const idx_t* ids;
  const IDSelector* sel;
  const IndexIvfPq* ivfpq;                  // added by tenann
  const float range_search_confidence = 0;  // added by tenann

  // wrapped result structure
  float radius;
  RangeQueryResult& rres;

  inline bool skip_entry(idx_t j) { return use_sel && !sel->is_member(ids[j]); }

  inline void add(idx_t j, float dis) {
    if constexpr (use_range_search_confidence) {
      /* The following lines are added by tenann */
      auto reconstruction_error = ivfpq->reconstruction_errors[key][j];
      // The result is valid if the distance lower bound <= radius.
      // Only works for L2 metric and ADC distance.
      auto lower_bound = std::abs(sqrtf(dis) - reconstruction_error * range_search_confidence);
      if (lower_bound <= radius) {
        idx_t id = ids ? ids[j] : lo_build(key, j);
        rres.add(dis, id);
      }
      /* End tenann. */
    } else {
      if (C::cmp(radius, dis)) {
        idx_t id = ids ? ids[j] : lo_build(key, j);
        rres.add(dis, id);
      }
    }
  }
};

/*****************************************************
 * Scaning the codes.
 * The scanning functions call their favorite precompute_*
 * function to precompute the tables they need.
 *****************************************************/
template <typename IDType, MetricType METRIC_TYPE, class PQDecoder>
struct IVFPQScannerT : QueryTables {
  const uint8_t* list_codes;
  const IDType* list_ids;
  size_t list_size;

  IVFPQScannerT(const IndexIvfPq& ivfpq, const IVFSearchParameters* params)
      : QueryTables(ivfpq, params) {
    assert(METRIC_TYPE == metric_type);
  }

  float dis0;

  void init_list(idx_t list_no, float coarse_dis, int mode) {
    this->key = list_no;
    this->coarse_dis = coarse_dis;

    if (mode == 2) {
      dis0 = precompute_list_tables();
    } else if (mode == 1) {
      dis0 = precompute_list_table_pointers();
    }
  }

  /*****************************************************
   * Scaning the codes: simple PQ scan.
   *****************************************************/

#ifdef __AVX2__
  /// Returns the distance to a single code.
  /// General-purpose version.
  template <class SearchResultType, typename T = PQDecoder>
  typename std::enable_if<!(std::is_same<T, PQDecoder8>::value),
                          float>::type inline distance_single_code(const uint8_t* code) const {
    PQDecoder decoder(code, pq.nbits);

    const float* tab = sim_table;
    float result = 0;

    for (size_t m = 0; m < pq.M; m++) {
      result += tab[decoder.decode()];
      tab += pq.ksub;
    }

    return result;
  }

  /// Returns the distance to a single code.
  /// Specialized AVX2 PQDecoder8 version.
  template <class SearchResultType, typename T = PQDecoder>
  typename std::enable_if<(std::is_same<T, PQDecoder8>::value),
                          float>::type inline distance_single_code(const uint8_t* code) const {
    float result = 0;

    size_t m = 0;
    const size_t pqM16 = pq.M / 16;

    const float* tab = sim_table;

    if (pqM16 > 0) {
      // process 16 values per loop

      const __m256i ksub = _mm256_set1_epi32(pq.ksub);
      __m256i offsets_0 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
      offsets_0 = _mm256_mullo_epi32(offsets_0, ksub);

      // accumulators of partial sums
      __m256 partialSum = _mm256_setzero_ps();

      // loop
      for (m = 0; m < pqM16 * 16; m += 16) {
        // load 16 uint8 values
        const __m128i mm1 = _mm_loadu_si128((const __m128i_u*)(code + m));
        {
          // convert uint8 values (low part of __m128i) to int32
          // values
          const __m256i idx1 = _mm256_cvtepu8_epi32(mm1);

          // add offsets
          const __m256i indices_to_read_from = _mm256_add_epi32(idx1, offsets_0);

          // gather 8 values, similar to 8 operations of tab[idx]
          __m256 collected = _mm256_i32gather_ps(tab, indices_to_read_from, sizeof(float));
          tab += pq.ksub * 8;

          // collect partial sums
          partialSum = _mm256_add_ps(partialSum, collected);
        }

        // move high 8 uint8 to low ones
        const __m128i mm2 = _mm_unpackhi_epi64(mm1, _mm_setzero_si128());
        {
          // convert uint8 values (low part of __m128i) to int32
          // values
          const __m256i idx1 = _mm256_cvtepu8_epi32(mm2);

          // add offsets
          const __m256i indices_to_read_from = _mm256_add_epi32(idx1, offsets_0);

          // gather 8 values, similar to 8 operations of tab[idx]
          __m256 collected = _mm256_i32gather_ps(tab, indices_to_read_from, sizeof(float));
          tab += pq.ksub * 8;

          // collect partial sums
          partialSum = _mm256_add_ps(partialSum, collected);
        }
      }

      // horizontal sum for partialSum
      const __m256 h0 = _mm256_hadd_ps(partialSum, partialSum);
      const __m256 h1 = _mm256_hadd_ps(h0, h0);

      // extract high and low __m128 regs from __m256
      const __m128 h2 = _mm256_extractf128_ps(h1, 1);
      const __m128 h3 = _mm256_castps256_ps128(h1);

      // get a final hsum into all 4 regs
      const __m128 h4 = _mm_add_ss(h2, h3);

      // extract f[0] from __m128
      const float hsum = _mm_cvtss_f32(h4);
      result += hsum;
    }

    //
    if (m < pq.M) {
      // process leftovers
      PQDecoder decoder(code + m, pq.nbits);

      for (; m < pq.M; m++) {
        result += tab[decoder.decode()];
        tab += pq.ksub;
      }
    }

    return result;
  }

#else
  /// Returns the distance to a single code.
  /// General-purpose version.
  template <class SearchResultType>
  inline float distance_single_code(const uint8_t* code) const {
    PQDecoder decoder(code, pq.nbits);

    const float* tab = sim_table;
    float result = 0;

    for (size_t m = 0; m < pq.M; m++) {
      result += tab[decoder.decode()];
      tab += pq.ksub;
    }

    return result;
  }
#endif

  /// version of the scan where we use precomputed tables.
  template <class SearchResultType>
  void scan_list_with_table(size_t ncode, const uint8_t* codes, SearchResultType& res) const {
    for (size_t j = 0; j < ncode; j++, codes += pq.code_size) {
      if (res.skip_entry(j)) {
        continue;
      }
      float dis = dis0 + distance_single_code<SearchResultType>(codes);
      res.add(j, dis);
    }
  }
};

/* We put as many parameters as possible in template. Hopefully the
 * gain in runtime is worth the code bloat.
 *
 * C is the comparator < or >, it is directly related to METRIC_TYPE.
 *
 * precompute_mode is how much we precompute (2 = precompute distance tables,
 * 1 = precompute pointers to distances, 0 = compute distances one by one).
 * Currently only 2 is supported
 *
 * use_sel: store or ignore the IDSelector
 */
template <MetricType METRIC_TYPE, class C, class PQDecoder, bool use_sel>
struct IVFPQScanner : IVFPQScannerT<Index::idx_t, METRIC_TYPE, PQDecoder>, InvertedListScanner {
  int precompute_mode;
  const IDSelector* sel;
  const IndexIvfPq* ivfpq;            // modifiled by tenann
  float range_search_confidence = 0;  // added by tenann

  IVFPQScanner(const IndexIvfPq& ivfpq, bool store_pairs, int precompute_mode,
               const IDSelector* sel)
      : IVFPQScannerT<Index::idx_t, METRIC_TYPE, PQDecoder>(ivfpq, nullptr),
        precompute_mode(precompute_mode),
        sel(sel),
        ivfpq(&ivfpq) {
    this->store_pairs = store_pairs;
  }

  void set_query(const float* query) override { this->init_query(query); }

  void set_list(idx_t list_no, float coarse_dis) override {
    this->list_no = list_no;
    this->init_list(list_no, coarse_dis, precompute_mode);
  }

  float distance_to_code(const uint8_t* code) const override {
    assert(precompute_mode == 2);
    float dis = this->dis0;
    const float* tab = this->sim_table;
    PQDecoder decoder(code, this->pq.nbits);

    for (size_t m = 0; m < this->pq.M; m++) {
      dis += tab[decoder.decode()];
      tab += this->pq.ksub;
    }
    return dis;
  }

  void scan_codes_range(size_t ncode, const uint8_t* codes, const idx_t* ids, float radius,
                        RangeQueryResult& rres) const override {
    /* The following lines are added by tenann */
    if (0 < range_search_confidence && range_search_confidence <= 1) {
      RangeSearchResults<C, use_sel, true> res = {
          /* key */ this->key,
          /* ids */ this->store_pairs ? nullptr : ids,
          /* sel */ this->sel,
          /* ivfpq */ this->ivfpq,                                      // added by tenann
          /* range_search_confidence */ this->range_search_confidence,  // added by tenann
          /* radius */ sqrtf(radius),  // modified by tenann (tenann uses squared root radius
                                       // instead)
          /* rres */ rres};

      if (this->polysemous_ht > 0) {
        assert(precompute_mode == 2);
        FAISS_THROW_MSG("polymous_ht is not supported for range search");
      } else if (precompute_mode == 2) {
        this->scan_list_with_table(ncode, codes, res);
      } else if (precompute_mode == 1) {
        FAISS_THROW_MSG("precompute_mode == 1 is not supported for range search");
      } else if (precompute_mode == 0) {
        FAISS_THROW_MSG("precompute_mode == 0 is not supported for range search");
      } else {
        FAISS_THROW_MSG("bad precomp mode");
      }

    } /* End tenann. */ else {
      RangeSearchResults<C, use_sel, false> res = {
          /* key */ this->key,
          /* ids */ this->store_pairs ? nullptr : ids,
          /* sel */ this->sel,
          /* ivfpq */ this->ivfpq,                                      // added by tenann
          /* range_search_confidence */ this->range_search_confidence,  // added by tenann
          /* radius */ radius,
          /* rres */ rres};

      if (this->polysemous_ht > 0) {
        assert(precompute_mode == 2);
        FAISS_THROW_MSG("polymous_ht is not supported for range search");
      } else if (precompute_mode == 2) {
        this->scan_list_with_table(ncode, codes, res);
      } else if (precompute_mode == 1) {
        FAISS_THROW_MSG("precompute_mode == 1 is not supported for range search");
      } else if (precompute_mode == 0) {
        FAISS_THROW_MSG("precompute_mode == 0 is not supported for range search");
      } else {
        FAISS_THROW_MSG("bad precomp mode");
      }
    }
  }
};

template <class PQDecoder, bool use_sel>
InvertedListScanner* get_InvertedListScanner1(const IndexIvfPq& index, bool store_pairs,
                                              const IDSelector* sel,
                                              float dynamic_range_search_confidence) {
  if (index.metric_type == METRIC_INNER_PRODUCT) {
    auto ret = new IVFPQScanner<METRIC_INNER_PRODUCT, CMin<float, idx_t>, PQDecoder, use_sel>(
        index, store_pairs, 2, sel);
    ret->range_search_confidence = dynamic_range_search_confidence;
    return ret;
  } else if (index.metric_type == METRIC_L2) {
    auto ret = new IVFPQScanner<METRIC_L2, CMax<float, idx_t>, PQDecoder, use_sel>(
        index, store_pairs, 2, sel);
    ret->range_search_confidence = dynamic_range_search_confidence;
    return ret;
  }
  return nullptr;
}

template <bool use_sel>
InvertedListScanner* get_InvertedListScanner2(const IndexIvfPq& index, bool store_pairs,
                                              const IDSelector* sel,
                                              float dynamic_range_search_confidence) {
  if (index.pq.nbits == 8) {
    return get_InvertedListScanner1<PQDecoder8, use_sel>(index, store_pairs, sel,
                                                         dynamic_range_search_confidence);
  } else if (index.pq.nbits == 16) {
    return get_InvertedListScanner1<PQDecoder16, use_sel>(index, store_pairs, sel,
                                                          dynamic_range_search_confidence);
  } else {
    return get_InvertedListScanner1<PQDecoderGeneric, use_sel>(index, store_pairs, sel,
                                                               dynamic_range_search_confidence);
  }
}

}  // anonymous namespace

InvertedListScanner* IndexIvfPq::custom_get_InvertedListScanner(
    bool store_pairs, const IDSelector* sel, float dynamic_range_search_confidence) const {
  if (sel) {
    return get_InvertedListScanner2<true>(*this, store_pairs, sel, dynamic_range_search_confidence);
  } else {
    return get_InvertedListScanner2<false>(*this, store_pairs, sel,
                                           dynamic_range_search_confidence);
  }
  return nullptr;
}

}  // namespace tenann
