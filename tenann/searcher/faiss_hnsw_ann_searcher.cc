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

#include "tenann/searcher/faiss_hnsw_ann_searcher.h"

#include "faiss/IndexHNSW.h"

namespace tenann {

void FaissHnswAnnSearcher::AnnSearch(PrimitiveSeqView query_vector, int k, int64_t* result_id) {
  T_DCHECK_NOTNULL(index_ref_);

  T_CHECK_EQ(index_ref_->index_type(), IndexType::kFaissHnsw);
  T_CHECK_EQ(query_vector.elem_type, PrimitiveType::kFloatType);

  auto index_hnsw = static_cast<faiss::IndexHNSW*>(index_ref_->index_raw());
  std::vector<float> distances(k);
  index_hnsw->search(1, reinterpret_cast<const float*>(query_vector.data), k, distances.data(),
                     result_id);
}

}  // namespace tenann