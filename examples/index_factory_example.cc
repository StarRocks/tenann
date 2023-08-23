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

#include "tenann/factory/index_factory.h"
#include "tenann/store/index_meta.h"

int main() {
  using namespace tenann;
  IndexMeta meta;

  // set meta values
  meta.SetMetaVersion(0);
  meta.SetIndexFamily(IndexFamily::kVectorIndex);
  meta.SetIndexType(IndexType::kFaissHnsw);
  meta.common_params()["dim"] = 128;
  meta.index_params()["efConstruction"] = 40;
  meta.search_params()["efSearch"] = 40;
  meta.extra_params()["comments"] = "my comments";

  try {
    auto index_builder = IndexFactory::CreateReaderFromMeta(meta);
  } catch (const char* e) {
    std::cerr << "Exception caught: " << e << "\n";
  }
}