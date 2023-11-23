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

#include <sys/time.h>

#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>

#include "tenann/common/logging.h"
#include "tenann/factory/ann_searcher_factory.h"
#include "tenann/factory/index_factory.h"
#include "tenann/store/index_meta.h"

std::vector<float> RandomVectors(uint32_t n, uint32_t dim, int seed = 0) {
  std::mt19937 rng(seed);
  std::vector<float> data(n * dim);
  std::uniform_real_distribution<> distrib;
  for (size_t i = 0; i < n * dim; i++) {
    data[i] = distrib(rng);
  }
  return data;
}

int main() {
  tenann::IndexMeta meta;

  // set meta values
  meta.SetMetaVersion(0);
  meta.SetIndexFamily(tenann::IndexFamily::kVectorIndex);
  meta.SetIndexType(tenann::IndexType::kFaissIvfPq);
  meta.common_params()["dim"] = 1024;
  meta.common_params()["is_vector_normed"] = false;
  meta.common_params()["metric_type"] = tenann::MetricType::kL2Distance;
  meta.index_params()["M"] = 8;
  meta.search_params()["nbits"] = 8;
  meta.extra_params()["comments"] = "my comments";

  // dimension of the vectors to index
  uint32_t d = 1024;
  // size of the database we plan to index
  size_t nb = 100000;
  // size of the query vectors we plan to test
  size_t nq = 10;
  // index save path
  std::string index_path = "/tmp/tenann_ivfpa_index";

  std::vector<int64_t> ids(nb);
  for (int i = 0; i < nb; i++) {
    ids[i] = i;
  }
  tenann::PrimitiveSeqView id_view = {.data = reinterpret_cast<uint8_t*>(ids.data()),
                                      .size = static_cast<uint32_t>(nb),
                                      .elem_type = tenann::PrimitiveType::kInt64Type};

  // generate data and query
  T_LOG(WARNING) << "Generating base vectors...";
  auto base = RandomVectors(nb, d);

  T_LOG(WARNING) << "Generating query vectors...";
  auto query = RandomVectors(nq, d, /*seed=*/1);

  // build and write index
  try {
    auto index_builder1 = tenann::IndexFactory::CreateBuilderFromMeta(meta);
    tenann::IndexWriterRef index_writer = tenann::IndexFactory::CreateWriterFromMeta(meta);
    index_builder1->SetIndexWriter(index_writer)
        .SetIndexCache(tenann::IndexCache::GetGlobalInstance())
        .Open(index_path);
    const int64_t step = 5000;
    auto* data = base.data();
    for (int64_t i = 0; i < nb; i += 5000) {
      T_LOG(WARNING) << "Adding data, offset " << i << "...";
      auto base_view = tenann::ArraySeqView{.data = reinterpret_cast<uint8_t*>(data),
                                            .dim = d,
                                            .size = static_cast<uint32_t>(step),
                                            .elem_type = tenann::PrimitiveType::kFloatType};
      index_builder1->Add({base_view});
      data += step;
    }
    T_LOG(WARNING) << "Flushing data...";
    index_builder1->Flush(true);
  } catch (tenann::Error& e) {
    std::cerr << "Exception caught: " << e.what() << "\n";
  }
}