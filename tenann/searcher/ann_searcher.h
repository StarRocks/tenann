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

#include <memory>

#include "tenann/common/seq_view.h"
#include "tenann/index/parameters.h"
#include "tenann/searcher/ann_filter.h"
#include "tenann/searcher/searcher.h"

namespace tenann {

#define ANN_SEARCHER_QUERY_COUNT (1)

class AnnSearcher : public Searcher<AnnSearcher> {
 public:
  explicit AnnSearcher(const IndexMeta& meta);
  virtual ~AnnSearcher() override;

  T_FORBID_MOVE(AnnSearcher);
  T_FORBID_COPY_AND_ASSIGN(AnnSearcher);

  /// ANN搜索接口，只返回k近邻的id
  virtual void AnnSearch(PrimitiveSeqView query_vector, int k, int64_t* result_id,
                         AnnFilter* ann_filter = nullptr) = 0;

  /// ANN搜索接口，同时返回k近邻的id和距离
  virtual void AnnSearch(PrimitiveSeqView query_vector, int k, int64_t* result_ids,
                         uint8_t* result_distances, AnnFilter* ann_filter = nullptr) = 0;

 protected:
  VectorIndexCommonParams common_params_;
};

using AnnSearcherRef = std::shared_ptr<AnnSearcher>;

}  // namespace tenann