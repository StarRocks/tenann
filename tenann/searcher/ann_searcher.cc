#include "ann_searcher.h"
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

#include "faiss/utils/distances.h"
#include "tenann/index/internal/metric_util.h"
#include "tenann/index/parameter_serde.h"
#include "tenann/util/distance_util.h"

namespace tenann {

AnnSearcher::AnnSearcher(const IndexMeta& meta) : Searcher<AnnSearcher>(meta) {
  FetchParameters(meta, &common_params_);
}

AnnSearcher::~AnnSearcher() = default;

void AnnSearcher::RangeSearch(PrimitiveSeqView query_vector, float range, int64_t limit,
                              ResultOrder result_order, std::vector<int64_t>* result_ids,
                              std::vector<float>* result_distances, const IdFilter* filter) {
  T_LOG(ERROR) << "range search not implemented";
}

void AnnSearcher::RangeSearch(PrimitiveSeqView query_vector, float range, int64_t limit,
                              ResultOrder result_order, std::vector<int64_t>* result_ids,
                              const IdFilter* id_filter) {
  std::vector<float> distanes;
  RangeSearch(query_vector, range, limit, result_order, result_ids, &distanes);
}

const float* AnnSearcher::PrepareCosineQuery(const float* query, size_t dim,
                                             std::vector<float>* scratch) const {
  // is_vector_normed=false means the index carries a normalizing pre-transform, which is applied
  // to the query as well -- there is nothing left to do here. For any other metric the query's
  // length is meaningful and must be left alone.
  if (common_params_.metric_type != MetricType::kCosineSimilarity ||
      !common_params_.is_vector_normed) {
    return query;
  }
  scratch->assign(query, query + dim);
  // Same routine faiss' NormalizationTransform uses, including its zero-norm guard.
  faiss::fvec_renorm_L2(dim, /*nx=*/1, scratch->data());
  return scratch->data();
}

void AnnSearcher::FinalizeScores(const int64_t* ids, float* scores, size_t n,
                                 faiss::MetricType physical_metric) const {
  if (common_params_.metric_type != MetricType::kCosineSimilarity) {
    return;
  }
  for (size_t i = 0; i < n; ++i) {
    if (ids != nullptr && ids[i] == -1) {
      continue;
    }
    if (NeedsL2ToCosine(static_cast<MetricType>(common_params_.metric_type), physical_metric)) {
      scores[i] = 1.0f - scores[i] * 0.5f;
    }
    scores[i] = std::clamp(scores[i], -1.0f, 1.0f);
  }
}

}  // namespace tenann
