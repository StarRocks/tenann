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
static constexpr const float radius = 15;
static constexpr const int nlist = 1;  // sqrt(nb);
static constexpr const int M = 32;
static constexpr const int nbits = 8;
static constexpr const int verbose = VERBOSE_INFO;

using namespace tenann;

IndexMeta PrepareMeta() {
  IndexMeta meta;
  meta.SetMetaVersion(0);
  meta.SetIndexFamily(tenann::IndexFamily::kVectorIndex);
  meta.SetIndexType(tenann::IndexType::kFaissIvfPq);
  meta.common_params()["dim"] = dim;
  meta.common_params()["is_vector_normed"] = false;
  meta.common_params()["metric_type"] = tenann::MetricType::kL2Distance;
  return meta;
}

json PrepareIndexParams(int nlist, int M, int nbits) {
  json index_params;
  index_params["nlist"] = nlist;
  index_params["M"] = M;
  index_params["nbits"] = nbits;
  return index_params;
}

int main(int argc, char const* argv[]) {
  auto base = RandomVectors(nb, dim, 0);
  auto query = RandomVectors(nq, dim, 1);

  auto meta = PrepareMeta();
  auto index_params = PrepareIndexParams(nlist, M, nbits);

  RangeSearchEvaluator eval("range_eval_exmaple", meta, ".");
  eval.SetVerboseLevel(verbose).SetDim(dim).SetBase(nb, base.data());
  eval.BuildIndexIfNotExists(index_params, true);
  eval.OpenSearcher();
  eval.CloseSearcher();
}