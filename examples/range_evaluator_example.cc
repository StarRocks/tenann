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
#include "tenann/bench/range_search_evaluator.h"
#include "tenann/common/logging.h"
#include "tenann/common/typed_seq_view.h"
#include "tenann/index/internal/index_ivfpq.h"
#include "tenann/util/random.h"
#include "tenann/util/runtime_profile_macros.h"
#include "tenann/util/threads.h"

static constexpr const int dim = 1024;
static constexpr const int nb = 1000000;
static constexpr const int nq = 100;
static constexpr const int verbose = VERBOSE_INFO;

using namespace tenann;

IndexMeta PrepareHnswMeta(MetricType metric_type) {
  IndexMeta meta;
  meta.SetMetaVersion(0);
  meta.SetIndexFamily(tenann::IndexFamily::kVectorIndex);
  meta.SetIndexType(tenann::IndexType::kFaissIvfPq);
  meta.common_params()["dim"] = dim;
  meta.common_params()["is_vector_normed"] = false;
  meta.common_params()["metric_type"] = metric_type;
  meta.index_writer_options()["write_index_cache"] = true;
  return meta;
}

json PrepareHnswParams(int M, int efConstruction) {
  json index_params;
  index_params["M"] = M;
  index_params["efConstruction"] = efConstruction;
  return index_params;
}

RangeQuerySet GenQuerySet(const std::vector<float>& query_list, int64_t nq, int dim,
                          float distance_threshold, int64_t limit) {
  RangeQuerySet query_set;
  query_set.nq = nq;
  query_set.query = query_list.data();
  for (int64_t i = 0; i < nq; i++) {
    query_set.limit_list.push_back(limit);
    query_set.distance_threshold_list.push_back(distance_threshold);
  }
  return query_set;
}

void Eval(MetricType metric_type, float threshold, int64_t limit, const std::vector<float>& base,
          const std::vector<float>& query) {
  auto query_set = GenQuerySet(query, nq, dim, threshold, limit);

  auto meta = PrepareHnswMeta(metric_type);
  auto index_params = PrepareHnswParams(16, 500);
  SetVLogLevel(verbose);

  RangeSearchEvaluator evaluator(
      metric_type == MetricType::kL2Distance ? "range_eval_exmaple_l2" : "range_eval_exmaple_cos",
      meta, ".");
  evaluator
      .SetMetricType(metric_type)  //
      .SetDim(dim)                 //
      .SetBase(nb, base.data())    //
      .SetQuery(nq, query_set);
  std::vector<json> search_param_list = {
      {{"efSearch", 10}},   //
      {{"efSearch", 20}},   //
      {{"efSearch", 40}},   //
      {{"efSearch", 80}},   //
      {{"efSearch", 100}},  //
      {{"efSearch", 200}}   //
  };

  evaluator
      .BuildIndexIfNotExists(index_params)  //
      .Evaluate(search_param_list);
}

int main(int argc, char const* argv[]) {
  auto base = RandomVectors(nb, dim, 0);
  auto query = RandomVectors(nq, dim, 1);

  std::cout << "======================= CosineSimlarity >= 0.8 limit 10 =======================\n";
  Eval(MetricType::kCosineSimilarity, 0.8, 10, base, query);

  std::cout << "======================= CosineSimlarity >= 0.8 =======================\n";
  Eval(MetricType::kCosineSimilarity, 0.8, -1, base, query);

  std::cout << "======================= l2_distance <= 10 limit 10 =======================\n";
  Eval(MetricType::kL2Distance, 12, 10, base, query);

  std::cout << "======================= l2_distance <= 10 =======================\n";
  Eval(MetricType::kL2Distance, 12, -1, base, query);
}