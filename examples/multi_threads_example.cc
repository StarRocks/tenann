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
#include "tenann/util/bruteforce.h"
#include "tenann/util/pretty_printer.h"
#include "tenann/util/runtime_profile_macros.h"
#include "tenann/util/threads.h"

std::vector<float> RandomVectors(uint32_t n, uint32_t dim, int seed = 0) {
  std::mt19937 rng(seed);
  std::vector<float> data(n * dim);
  std::uniform_real_distribution<> distrib;
  for (size_t i = 0; i < n * dim; i++) {
    data[i] = distrib(rng);
  }
  return data;
}
// dimension of the vectors to index
constexpr const uint32_t d = 1024;
// size of the database we plan to index
constexpr const size_t nb = 100000;
// size of the query vectors we plan to test
constexpr const size_t nq = 10;
constexpr const auto metric = tenann::MetricType::kL2Distance;

auto GetHnswMeta() {
  tenann::IndexMeta meta;
  // set meta values
  meta.SetMetaVersion(0);
  meta.SetIndexFamily(tenann::IndexFamily::kVectorIndex);
  meta.SetIndexType(tenann::IndexType::kFaissHnsw);
  meta.common_params()["dim"] = 128;
  meta.common_params()["is_vector_normed"] = false;
  meta.common_params()["metric_type"] = metric;
  meta.index_params()["efConstruction"] = 40;
  meta.index_params()["M"] = 128;
  meta.search_params()["efSearch"] = 40;
  meta.extra_params()["comments"] = "my comments";
  return meta;
}

auto GetIvfPqMeta() {
  tenann::IndexMeta meta;
  // set meta values
  meta.SetMetaVersion(0);
  meta.SetIndexFamily(tenann::IndexFamily::kVectorIndex);
  meta.SetIndexType(tenann::IndexType::kFaissIvfPq);
  meta.common_params()["dim"] = d;
  meta.common_params()["is_vector_normed"] = false;
  meta.common_params()["metric_type"] = metric;
  meta.index_params()["nlist"] = 256;
  meta.index_params()["M"] = d / 4;
  meta.index_params()["nbits"] = 8;
  meta.search_params()["nprobe"] = 128;
  meta.extra_params()["comments"] = "my comments";
  return meta;
}

int main() {
  auto meta = GetIvfPqMeta();

  // index save path
  std::string index_path = "/tmp/faiss_index";

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
  auto base_col = tenann::ArraySeqView{.data = reinterpret_cast<uint8_t*>(base.data()),
                                       .dim = d,
                                       .size = static_cast<uint32_t>(nb),
                                       .elem_type = tenann::PrimitiveType::kFloatType};
  // build and write index
  auto index_builder1 = tenann::IndexFactory::CreateBuilderFromMeta(meta);
  auto index_builder2 = tenann::IndexFactory::CreateBuilderFromMeta(meta);

  auto profile = std::make_unique<tenann::RuntimeProfile>("root");
  auto single_thread_timer = T_ADD_TIMER(profile, "single");
  auto multi_thread_timer = T_ADD_TIMER(profile, "multi");

  T_LOG(WARNING) << "Build with 4 threads...";
  tenann::OmpSetNumThreads(4);
  {
    T_SCOPED_TIMER(single_thread_timer);
    index_builder1->Open(index_path)
        .Add({base_col});
  }

  T_LOG(WARNING) << "Build with 12 threads...";
  tenann::OmpSetNumThreads(12);
  {
    T_SCOPED_TIMER(multi_thread_timer);
    index_builder2->Open(index_path)
        .Add({base_col});
  }

  std::cout << "Build with 4 threads: "
            << tenann::PrettyPrinter::print(single_thread_timer->value(), tenann::TUnit::TIME_NS)
            << "\n";

  std::cout << "Build with 12 threads: "
            << tenann::PrettyPrinter::print(multi_thread_timer->value(), tenann::TUnit::TIME_NS)
            << "\n";
}