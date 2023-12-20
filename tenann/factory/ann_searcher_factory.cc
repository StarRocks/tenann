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

#include "tenann/factory/ann_searcher_factory.h"
#include "tenann/searcher/faiss_hnsw_ann_searcher.h"
#include "tenann/searcher/faiss_ivf_pq_ann_searcher.h"
#include "tenann/common/logging.h"

namespace tenann {

std::shared_ptr<AnnSearcher> AnnSearcherFactory::CreateSearcherFromMeta(const IndexMeta& meta) {
  if (meta.index_type() == IndexType::kFaissHnsw) {
    return std::make_unique<FaissHnswAnnSearcher>(meta);
  } else if(meta.index_type() == IndexType::kFaissIvfPq) {
    return std::make_unique<FaissIvfPqAnnSearcher>(meta);
  } else {
    T_LOG(ERROR) << "Unsupported index type: " << static_cast<int>(meta.index_type());
  }
}

}  // namespace tenann