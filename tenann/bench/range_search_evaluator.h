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

#include "fmt/format.h"
#include "tenann/bench/evaluator.h"
#include "tenann/common/json.h"
#include "tenann/factory/ann_searcher_factory.h"
#include "tenann/factory/index_factory.h"
#include "tenann/index/index_str.h"
#include "tenann/searcher/ann_searcher.h"
#include "tenann/util/filesystem.h"

namespace tenann {

struct RangeQuerySet {
  const float* query;
  int64_t nq;
  std::vector<float> radius_list;
  std::vector<int64_t> limit_list;
  std::vector<AnnSearcher::ResultOrder> orders;

  RangeQuerySet() = default;

  RangeQuerySet(const float* query, int64_t nq, const std::vector<float>& radius_list,
                const std::vector<int64_t> limit_list,
                const std::vector<AnnSearcher::ResultOrder>& orders)
      : query(query), nq(nq), radius_list(radius_list), limit_list(limit_list), orders(orders) {}
};

struct RangeSearchMetrics {
  double latency;
  double qps;
  double recall;
  double precision;
  int64_t result_cardinality;
  int64_t nq;

  std::string Str() {
    json doc;
    doc["qps"] = qps;
    doc["recall"] = recall;
    doc["precision"] = qps;
    doc["result_cardinality"] = result_cardinality;
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

  QueryResultList ComputeGroundTruth() override {}

  EvaluationMetrics EvaluateSingleQuery(int64_t i, const json& search_params) override {}

  EvaluationMetrics CreateEvaluationMetrics() override {
    return EvaluationMetrics{
        .latency = 0, .qps = 0, .recall = 0, .precision = 0, .result_cardinality = 0, .nq = 0};
  }

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
