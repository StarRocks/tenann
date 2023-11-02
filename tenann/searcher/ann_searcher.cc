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

#include "tenann/index/parameter_serde.h"

namespace tenann {

AnnSearcher::AnnSearcher(const IndexMeta& meta) : Searcher<AnnSearcher>(meta) {
  FetchParameters(meta, &common_params_);
}

AnnSearcher::~AnnSearcher() = default;

void AnnSearcher::RangeSearch(PrimitiveSeqView query_vector, float range, int64_t limit,
                              ResultOrder result_order, std::vector<int64_t>* result_ids,
                              std::vector<float>* result_distances) {
  T_LOG(ERROR) << "range search not implemented";
}

}  // namespace tenann