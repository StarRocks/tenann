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

#include "tenann/common/json.h"
#include "tenann/common/logging.h"
#include "tenann/store/index_meta.h"

#define RETURN_SELF_AFTER(stmt) stmt return *this;

#define VERBOSE_CRITICAL (0)
#define VERBOSE_INFO (1)
#define VERBOSE_DEBUG (2)
#define VLOG_CRITICAL(level) T_LOG_IF(INFO, level >= VERBOSE_CRITICAL)
#define VLOG_INFO(level) T_LOG_IF(INFO, level >= VERBOSE_INFO)
#define VLOG_DEBUG(level) T_LOG_IF(INFO, level >= VERBOSE_DEBUG)

namespace tenann {

template <typename QuerySet, typename EvaluationMetrics>
class Evaluator {
 public:
  /// @brief using this line to trigger an compiler error if `EvaluationMetrics` does not
  /// implement the method `Str()`
  using Str_ = decltype(EvaluationMetrics().Str());

  using Self = Evaluator;
  using RowidColumn = std::vector<int64_t>;
  using DistanceColumn = std::vector<float>;
  using QueryResult = std::pair<RowidColumn, DistanceColumn>;
  using QueryResultList = std::vector<QueryResult>;

  Evaluator() = default;
  virtual ~Evaluator() = default;

  Self& SetMetricType(MetricType metric_type) { RETURN_SELF_AFTER(metric_type_ = metric_type;) }

  Self& SetVerboseLevel(int level) { RETURN_SELF_AFTER(verbose_level_ = level;) }

  Self& SetDim(int dim) { RETURN_SELF_AFTER(dim_ = dim;) }

  Self& SetBase(int64_t nb, const float* base) { RETURN_SELF_AFTER(base_ = base; nb_ = nb;) }

  Self& SetQuery(int64_t nq, const QuerySet& query_set) {
    nq_ = nq;
    query_set_ = query_set;
    return *this;
  }

  std::vector<std::tuple<json, json, EvaluationMetrics>> Evaluate(
      const std::vector<json>& search_params_list) {
    OpenSearcher();

    if (ground_truth_.empty()) {
      ground_truth_ = ComputeGroundTruth();
    }

    std::vector<std::tuple<json, json, EvaluationMetrics>> evaluation_results;
    for (const auto& search_params : search_params_list) {
      VLOG_INFO(verbose_level_) << "Evaluating params: " << search_params << " ...";
      EvaluationMetrics global_metrics = CreateEvaluationMetrics();
      for (int64_t i = 0; i < nq_; i++) {
        auto query_metrics = EvaluateSingleQuery(i, search_params);

        VLOG_DEBUG(verbose_level_)
            << "Evaluation results of query " << i << ": " << query_metrics.Str();
        MergeEvaluationMetrics(global_metrics, query_metrics);
      }

      FinalizeEvaluationMetrics(global_metrics);
      VLOG_INFO(verbose_level_) << "Evaluaton results: " << global_metrics.Str();
      evaluation_results.emplace_back(index_params_, search_params, global_metrics);
    };

    CloseSearcher();
    return evaluation_results;
  }

  virtual Self& BuildIndexIfNotExists(const json& index_params, bool force_rebuild = false) = 0;
  virtual Self& OpenSearcher() { return *this; };
  virtual Self& CloseSearcher() { return *this; };

 protected:
  virtual QueryResultList ComputeGroundTruth() = 0;
  virtual EvaluationMetrics EvaluateSingleQuery(int64_t i, const json& search_params) = 0;

  virtual EvaluationMetrics CreateEvaluationMetrics() = 0;
  virtual void MergeEvaluationMetrics(EvaluationMetrics& dst, const EvaluationMetrics& src) = 0;
  virtual void FinalizeEvaluationMetrics(EvaluationMetrics& dst) = 0;

 protected:
  int verbose_level_ = VERBOSE_CRITICAL;
  MetricType metric_type_;
  int dim_ = -1;
  const float* base_ = nullptr;
  int64_t nb_ = 0;
  int64_t nq_ = 0;

  QuerySet query_set_;
  QueryResultList ground_truth_;
  json index_params_;
  IndexMeta index_meta_;
};
}  // namespace tenann