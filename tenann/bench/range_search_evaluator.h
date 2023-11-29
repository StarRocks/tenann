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

#pragma once

#include <vector>

#include "faiss/impl/AuxIndexStructures.h"
#include "fmt/format.h"
#include "tenann/bench/evaluator.h"
#include "tenann/common/json.h"
#include "tenann/common/macros.h"
#include "tenann/factory/ann_searcher_factory.h"
#include "tenann/factory/index_factory.h"
#include "tenann/index/index_str.h"
#include "tenann/searcher/ann_searcher.h"
#include "tenann/util/bruteforce.h"
#include "tenann/util/filesystem.h"
#include "tenann/util/runtime_profile_macros.h"
namespace tenann {

struct RangeQuerySet {
  const float* query;
  int64_t nq;
  std::vector<float> distance_threshold_list;
  std::vector<int64_t> limit_list;

  RangeQuerySet() = default;

  RangeQuerySet(const float* query, int64_t nq, const std::vector<float>& distance_threshold_list,
                const std::vector<int64_t> limit_list)
      : query(query),
        nq(nq),
        distance_threshold_list(distance_threshold_list),
        limit_list(limit_list) {}
};

struct RangeSearchMetrics {
  double latency = 0;
  double qps = 0;
  double recall = 0;
  double precision = 0;
  int64_t result_cardinality = 0;
  int64_t nq = 0;

  std::string Str() {
    json doc = {{"latency", latency},
                {"qps", qps},
                {"recall", recall},
                {"precision", precision},
                {"result_cardinality", result_cardinality},
                {"nq", nq}};
    return doc.dump();
  }
};

class RangeSearchEvaluator : public Evaluator<RangeQuerySet, RangeSearchMetrics> {
 public:
  using Self = Evaluator<RangeQuerySet, RangeSearchMetrics>;
  using EvaluationMetrics = RangeSearchMetrics;

  RangeSearchEvaluator(const std::string& evaluator_name, const IndexMeta& index_meta,
                       const std::string& index_save_dir)
      : Evaluator<RangeQuerySet, RangeSearchMetrics>(),
        index_save_dir_(index_save_dir),
        evaluator_name_(evaluator_name) {
    index_meta_ = index_meta;
  }

  ~RangeSearchEvaluator() override {}

  Self& BuildIndexIfNotExists(const json& index_params, bool force_rebuild = false) override {
    index_params_ = index_params;
    index_meta_.index_params() = index_params;

    auto index_str = IndexStr(index_meta_);
    auto index_path = fmt::format("{}/{}_{}", index_save_dir_, evaluator_name_, index_str);

    VLOG_CRITICAL(verbose_level_) << "Start building index: " << index_path << " ...";
    if (FileExists(index_path) && !force_rebuild) {
      VLOG_CRITICAL(verbose_level_)
          << "Index already exists: " << index_path << ", skip index building.";
      return *this;
    }

    std::shared_ptr<IndexWriter> writer = IndexFactory::CreateWriterFromMeta(index_meta_);
    auto builder = IndexFactory::CreateBuilderFromMeta(index_meta_);

    builder
        ->SetIndexWriter(writer)                         //
        .SetIndexCache(IndexCache::GetGlobalInstance())  //
        .Open(index_path);

    ArraySeqView base_view{.data = reinterpret_cast<const uint8_t*>(base_),
                           .dim = dim_,
                           .size = nb_,
                           .elem_type = PrimitiveType::kFloatType};

    builder->Add({base_view});
    builder->Flush(true);
    builder->Close();

    VLOG_CRITICAL(verbose_level_) << "Done index building: " << index_path;
    return *this;
  }

  Self& OpenSearcher() override {
    auto index_str = IndexStr(index_meta_);
    auto index_path = fmt::format("{}/{}_{}", index_save_dir_, evaluator_name_, index_str);

    std::shared_ptr<IndexReader> reader = IndexFactory::CreateReaderFromMeta(index_meta_);
    searcher_ = AnnSearcherFactory::CreateSearcherFromMeta(index_meta_);
    searcher_
        ->SetIndexReader(reader)  //
        .SetIndexCache(IndexCache::GetGlobalInstance())
        .ReadIndex(index_path);
    return *this;
  }

  Self& CloseSearcher() override { return *this; }

  QueryResultList ComputeGroundTruth() override {
    T_CHECK(query_set_.nq != 0 && query_set_.query != nullptr);

    auto base_view = tenann::ArraySeqView{.data = reinterpret_cast<const uint8_t*>(base_),
                                          .dim = dim_,
                                          .size = nb_,
                                          .elem_type = PrimitiveType::kFloatType};

    T_LOG(INFO) << "Computing ground truth...";
    QueryResultList results(nq_);
    for (int i = 0; i < nq_; i++) {
      auto query_vector = tenann::PrimitiveSeqView{
          .data = reinterpret_cast<const uint8_t*>(query_set_.query + i * dim_),
          .size = dim_,
          .elem_type = PrimitiveType::kFloatType};
      util::BruteForceRangeSearch(metric_type_, dim_, base_view, nullptr, nullptr, query_vector,
                                  query_set_.distance_threshold_list[i], query_set_.limit_list[i],
                                  metric_type_ == MetricType::kL2Distance
                                      ? AnnSearcher::ResultOrder::kAscending
                                      : AnnSearcher::ResultOrder::kDescending,
                                  &results[i].first, &results[i].second);
    }

    T_LOG(INFO) << "Done computing ground truth.";

    return results;
  }

  EvaluationMetrics EvaluateSingleQuery(int64_t i, const json& search_params) override {
    T_CHECK(!ground_truth_.empty()) << "missing ground truth";
    searcher_->SetSearchParams(search_params);
    auto query_vector = tenann::PrimitiveSeqView{
        .data = reinterpret_cast<const uint8_t*>(query_set_.query + i * dim_),
        .size = dim_,
        .elem_type = PrimitiveType::kFloatType};

    std::vector<int64_t> result_ids;
    int64_t latency;
    {
      T_SCOPED_RAW_TIMER(&latency);
      searcher_->RangeSearch(
          query_vector, query_set_.distance_threshold_list[i], query_set_.limit_list[i],
          metric_type_ == MetricType::kL2Distance ? AnnSearcher::ResultOrder::kAscending
                                                  : AnnSearcher::ResultOrder::kDescending,
          &result_ids);
    }
    auto [precision, recall, result_cardinality] = ReportSingle(ground_truth_[i].first, result_ids);

    RangeSearchMetrics metrics;
    metrics.nq += 1;
    metrics.latency = double(latency) / 1000 / 1000 / 1000;
    metrics.precision = precision;
    metrics.recall = recall;
    metrics.result_cardinality = result_cardinality;
    // std::cout << "Precision:" << precision << ",Recall:" << recall << "\n";
    // std::cout << "Metrics for query " << i << " " << metrics.Str() << "\n";
    return metrics;
  }

  /// Precision, recall, result_cardinality.
  /// If the ground-truth result set is empty, the recall is set to 1,
  /// and the precision is set to 1 / (cardinality of search results).
  static std::tuple<double, double, double> ReportSingle(const std::vector<int64_t> gt_ids,
                                                         const std::vector<int64_t> result_ids) {
    double precision, recall, result_cardinality;
    result_cardinality = result_ids.size();
    double gt_cardinality = gt_ids.size();

    if (gt_cardinality == 0) {
      precision = result_cardinality == 0 ? 1.0 : 1 / result_cardinality;
      recall = 1.0;
      return std::make_tuple(precision, recall, result_cardinality);
    }

    std::set<int64_t> gt_set;
    for (int i = 0; i < gt_ids.size(); i++) {
      gt_set.insert(gt_ids[i]);
    }

    int64_t hit = 0;
    for (int i = 0; i < result_ids.size(); i++) {
      if (gt_set.find(result_ids[i]) != gt_set.end()) {
        hit += 1;
      }
    }

    // we have gt_carnality > 0 here
    recall = hit / gt_cardinality;
    precision = result_cardinality == 0 ? 0 : hit / result_cardinality;
    return std::make_tuple(precision, recall, result_cardinality);
  }

  EvaluationMetrics CreateEvaluationMetrics() override { return RangeSearchMetrics(); }

  void MergeEvaluationMetrics(EvaluationMetrics& dst, const EvaluationMetrics& src) override {
    dst.latency += src.latency;
    dst.recall += src.recall;
    dst.precision += src.precision;
    dst.result_cardinality += src.result_cardinality;
    dst.nq += src.nq;
  }

  void FinalizeEvaluationMetrics(EvaluationMetrics& dst) override {
    dst.latency /= dst.nq;
    dst.qps = dst.nq / dst.latency;
    dst.recall /= dst.nq;
    dst.precision /= dst.nq;
    dst.result_cardinality /= dst.nq;
  }

 protected:
  std::unique_ptr<AnnSearcher> searcher_;
  std::string index_save_dir_;
  std::string evaluator_name_;
};

}  // namespace tenann
