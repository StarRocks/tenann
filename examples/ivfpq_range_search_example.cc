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

#include <iostream>
#include <random>
#include <vector>

#include "faiss/IndexFlat.h"
#include "faiss/IndexIVFPQ.h"
#include "faiss/impl/AuxIndexStructures.h"
#include "faiss/utils/distances.h"
#include "tenann/common/logging.h"
#include "tenann/common/typed_seq_view.h"
#include "tenann/index/internal/IndexIVFPQ.h"
#include "tenann/index/internal/custom_ivfpq.h"
#include "tenann/util/runtime_profile_macros.h"
#include "tenann/util/threads.h"

#define RETURN_SELF_AFTER(stmt) stmt return *this;

template <class IVFPQ = faiss::IndexIVFPQ>
class RangeSearchEvaluator {
 public:
  using Self = RangeSearchEvaluator;

  Self& SetVerbose(bool verbose) { RETURN_SELF_AFTER(verbose_ = verbose;) }

  Self& SetDim(int dim) { RETURN_SELF_AFTER(dim_ = dim;) }

  Self& SetBase(int64_t nb, const float* base) { RETURN_SELF_AFTER(base_ = base; nb_ = nb;) }

  Self& SetQuery(int64_t nq, const float* query) { RETURN_SELF_AFTER(query_ = query; nq_ = nq;) }

  Self& SetRadius(float radius) { RETURN_SELF_AFTER(radius_ = radius;) }

  Self& SetNList(int n_list) { RETURN_SELF_AFTER(this->n_list_ = n_list;) }

  Self& SetM(int m) { RETURN_SELF_AFTER(this->m_ = m;) }

  Self& SetNBits(int nbits) { RETURN_SELF_AFTER(this->nbits_ = nbits;) }

  Self& BuildIndex() {
    T_LOG(INFO) << "Building index...";

    coarse_quantizer_ = std::make_unique<faiss::IndexFlatL2>(dim_);
    ivfpq_ = std::make_unique<IVFPQ>(coarse_quantizer_.get(), dim_, n_list_, m_, nbits_);
    ivfpq_->train(nb_, base_);
    ivfpq_->add(nb_, base_);

    T_LOG(INFO) << "Done building index.";
    return *this;
  }

  Self& ComputeGroundTruth() {
    T_LOG(INFO) << "Computing ground truth...";

    T_CHECK(dim_ > 0);
    ground_truth_ = std::make_unique<faiss::RangeSearchResult>(nq_);
    faiss::range_search_L2sqr(query_, base_, dim_, nq_, nb_, radius_, ground_truth_.get());

    if (verbose_) {
      PrintRangeSearchResults(ground_truth_.get());
    }

    T_LOG(INFO) << "Done computing ground truth.";
    return *this;
  }

  // nprobe, error_scale, QPS, precision, recall, result_cardinality
  using ResultItem = std::tuple<int64_t, float, double, double, double, double>;

  std::vector<ResultItem> Evaluate(const std::vector<int64_t> nprobe_list,
                                   const std::vector<float>& error_scale_list) {
    tenann::OmpSetNumThreads(1);
    std::vector<ResultItem> evaluation_results;

    for (auto nprobe : nprobe_list) {
      for (auto error_scale : error_scale_list) {
        ivfpq_->nprobe = nprobe;

        if constexpr (std::is_same_v<tenann::CustomIvfPq, IVFPQ>) {
          ivfpq_->error_scale = error_scale;
        }

        faiss::RangeSearchResult results(nq_);
        int64_t duration_ns = 0;
        {
          T_SCOPED_RAW_TIMER(&duration_ns);
          ivfpq_->range_search(nq_, query_, radius_, &results);
        }

        if (verbose_) {
          PrintRangeSearchResults(&results);
        }
        auto [precision, recall, result_cardinality] = Report(ground_truth_.get(), &results);
        double qps = double(nq_) / duration_ns * 1000 * 1000 * 1000;
        evaluation_results.emplace_back(nprobe, error_scale, qps, precision, recall,
                                        result_cardinality);

        if constexpr (std::is_same_v<faiss::IndexIVFPQ, IVFPQ>) {
          break;
        }
      }
    }

    if (verbose_) {
      T_LOG(INFO) << "Evaluation results:\n";
      PrintEvaluationResults(evaluation_results);
    }
    return evaluation_results;
  };

  static void PrintEvaluationResults(const std::vector<ResultItem>& results) {
    std::cout << "nprobe,error_scale,QPS,precision,recall,result_cardinality\n";
    for (auto [nprobe, error_scale, qps, precision, recall, result_cardinality] : results) {
      std::cout << std::setprecision(4) << nprobe << "," << error_scale << "," << qps << ","
                << precision << "," << recall << "," << result_cardinality << "\n";
    }
  }

  static void PrintRangeSearchResults(const faiss::RangeSearchResult* result) {
    for (int qi = 0; qi < result->nq; qi++) {
      std::cout << "****************************************\n";
      std::cout << "Result for query " << qi << ":\n";
      std::cout << "IDs:";
      for (int i = result->lims[qi]; i < result->lims[qi + 1]; i++) {
        std::cout << result->labels[i] << ",";
      }
      std::cout << "\n";
      std::cout << "Distances:";
      for (int i = result->lims[qi]; i < result->lims[qi + 1]; i++) {
        std::cout << result->distances[i] << std::setprecision(4) << ",";
      }
      std::cout << "\n";
    }
  }

  /// Compute average precision, recall, cardinality of search results.
  static std::tuple<double, double, double> Report(const faiss::RangeSearchResult* groudtruth,
                                                   const faiss::RangeSearchResult* result) {
    T_CHECK(groudtruth->nq == result->nq);
    auto nq = result->nq;

    double total_precision = 0, total_recall = 0, total_cardinality = 0;
    for (int qi = 0; qi < nq; qi++) {
      tenann::TypedSlice<int64_t> gt_ids{.data = groudtruth->labels + groudtruth->lims[qi],
                                         .size = groudtruth->lims[qi + 1] - groudtruth->lims[qi]};

      tenann::TypedSlice<int64_t> result_ids{.data = result->labels + result->lims[qi],
                                             .size = result->lims[qi + 1] - result->lims[qi]};

      auto [precision, recall, result_cardinality] = ReportSingle(gt_ids, result_ids);
      total_precision += precision;
      total_recall += recall;
      total_cardinality += result_cardinality;
    }

    auto precision = total_precision / nq;
    auto recall = total_recall / nq;
    auto cardinality = total_cardinality / nq;

    return std::make_tuple(precision, recall, cardinality);
  }

  /// Precision, recall, cc.
  /// If the ground-truth result set is empty, the recall is set to 1,
  /// and the precision is set to 1 / (cardinality of search results).
  static std::tuple<double, double, double> ReportSingle(tenann::TypedSlice<int64_t> gt_ids,
                                                         tenann::TypedSlice<int64_t> result_ids) {
    double precision, recall, result_cardinality;
    result_cardinality = result_ids.size;
    double gt_cardinality = gt_ids.size;

    if (gt_cardinality == 0) {
      precision = result_cardinality == 0 ? 1.0 : 1 / result_cardinality;
      recall = 1.0;
      return std::make_tuple(precision, recall, result_cardinality);
    }

    std::set<int64_t> gt_set;
    for (int i = 0; i < gt_ids.size; i++) {
      gt_set.insert(gt_ids.data[i]);
    }

    int64_t hit = 0;
    for (int i = 0; i < result_ids.size; i++) {
      if (gt_set.find(result_ids.data[i]) != gt_set.end()) {
        hit += 1;
      }
    }

    // we have gt_carnality > 0 here
    recall = hit / gt_cardinality;
    precision = result_cardinality == 0 ? 0 : hit / result_cardinality;
    return std::make_tuple(precision, recall, result_cardinality);
  }

 private:
  bool verbose_ = true;
  int dim_ = -1;
  const float* base_ = nullptr;
  size_t nb_ = 0;
  const float* query_ = nullptr;
  float radius_ = 0;
  size_t nq_ = 0;

  int n_list_ = 1;
  int m_ = 1;
  int nbits_ = 8;

  std::unique_ptr<faiss::IndexFlatL2> coarse_quantizer_ = nullptr;
  std::unique_ptr<IVFPQ> ivfpq_ = nullptr;
  std::unique_ptr<faiss::RangeSearchResult> ground_truth_ = nullptr;
};

std::vector<float> RandomVectors(uint32_t n, uint32_t dim, int seed = 0) {
  std::mt19937 rng(seed);
  std::vector<float> data(n * dim);
  std::uniform_real_distribution<> distrib;
  for (size_t i = 0; i < n * dim; i++) {
    data[i] = distrib(rng);
  }
  return data;
}

int main(int argc, char const* argv[]) {
  const int dim = 128;
  const int nb = 100000;
  const int nq = 1000;
  const float radius = 15;
  const int nlist = 1;  // sqrt(nb);
  const int M = 32;
  const int nbits = 8;
  const bool verbose = false;

  auto base = RandomVectors(nb, dim, 0);
  auto query = RandomVectors(nq, dim, 1);

  std::vector<int64_t> nprobe_list = {nlist};
  std::vector<float> error_scale_list = {0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 1};

  tenann::OmpSetNumThreads(16);
  RangeSearchEvaluator<tenann::CustomIvfPq> evaluator1;
  evaluator1.SetVerbose(verbose)
      .SetBase(nb, base.data())    //
      .SetQuery(nq, query.data())  //
      .SetDim(dim)                 //
      .SetNList(nlist)             //
      .SetM(M)                     //
      .SetNBits(nbits)             //
      .BuildIndex();
  auto result1 = evaluator1
                     .SetRadius(radius)     //
                     .ComputeGroundTruth()  //
                     .Evaluate(nprobe_list, error_scale_list);

  tenann::OmpSetNumThreads(16);
  RangeSearchEvaluator<faiss::IndexIVFPQ> evaluator2;
  evaluator2.SetVerbose(verbose)
      .SetBase(nb, base.data())    //
      .SetQuery(nq, query.data())  //
      .SetDim(dim)                 //
      .SetNList(nlist)             //
      .SetM(M)                     //
      .SetNBits(nbits)             //
      .BuildIndex();
  auto result2 = evaluator2
                     .SetRadius(radius)     //
                     .ComputeGroundTruth()  //
                     .Evaluate(nprobe_list, error_scale_list);

  RangeSearchEvaluator<>::PrintEvaluationResults(result1);
  RangeSearchEvaluator<>::PrintEvaluationResults(result2);
  return 0;
}
