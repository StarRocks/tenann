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

#include "tenann/builder/faiss_index_builder.h"
#include "tenann/common/parameters.h"

namespace faiss {
class IndexHNSW;
}

namespace tenann {

class FaissHnswIndexBuilder final : public FaissIndexBuilder {
 public:
  using FaissIndexBuilder::FaissIndexBuilder;
  virtual ~FaissHnswIndexBuilder();

  T_FORBID_COPY_AND_ASSIGN(FaissHnswIndexBuilder);
  T_FORBID_MOVE(FaissHnswIndexBuilder);

 protected:
  IndexRef InitIndex() override;

 private:
  void InitParameters(const IndexMeta& meta);
  std::string FactoryString();
  faiss::IndexHNSW* FetchHnsw(faiss::Index* index);

  FaissHnswIndexParams index_params_;
  FaissHnswSearchParams search_params_;
};

}  // namespace tenann
