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

#include "tenann/searcher/ann_searcher.h"

namespace tenann {

class FaissHnswAnnSearcher : public AnnSearcher {
 public:
  explicit FaissHnswAnnSearcher(const IndexMeta& meta);
  virtual ~FaissHnswAnnSearcher();

  T_FORBID_MOVE(FaissHnswAnnSearcher);
  T_FORBID_COPY_AND_ASSIGN(FaissHnswAnnSearcher);

  /// ANN搜索接口，只返回k近邻的id
  void AnnSearch(PrimitiveSeqView query_vector, int k, int64_t* result_id,
                 const IdFilter* id_filter = nullptr) override;

  void AnnSearch(PrimitiveSeqView query_vector, int k, int64_t* result_ids,
                 uint8_t* result_distances, const IdFilter* id_filter = nullptr) override;

 protected:
  void OnSearchParamItemChange(const std::string& key, const json& value) override;

  void OnSearchParamsChange(const json& value) override;

  void OnIndexLoaded() override;

 private:
  FaissHnswSearchParams search_params_;
  const void* faiss_id_map_;
  const void* faiss_transform_;
  const void* faiss_hnsw_;
};

}  // namespace tenann