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

void PrintResults(const std::vector<int64_t>& result_ids,
                  const std::vector<float>& result_distances, int nq, int k) {
  // search index
  for (int i = 0; i < nq; i++) {
    std::cout << "Result of query " << i << ": ";
    for (int j = 0; j < k; j++) {
      std::cout << result_ids[i * k + j] << ",";
    }
    std::cout << "\n";

    for (int j = 0; j < k; j++) {
      std::cout << result_distances[i * k + j] << ",";
    }
    std::cout << "\n";
  }
  std::cout << "-------------------------------------------------------------------\n";
}

int main() {
  tenann::OmpSetNumThreads(8);
  tenann::IndexMeta meta;

  auto metric = tenann::MetricType::kCosineSimilarity;
  // set meta values
  meta.SetMetaVersion(0);
  meta.SetIndexFamily(tenann::IndexFamily::kVectorIndex);
  meta.SetIndexType(tenann::IndexType::kFaissHnsw);
  meta.common_params()["dim"] = 128;
  meta.common_params()["is_vector_normed"] = false;
  meta.common_params()["metric_type"] = metric;
  meta.index_params()["efConstruction"] = 500;
  meta.index_params()["M"] = 128;
  meta.search_params()["efSearch"] = 80;
  meta.extra_params()["comments"] = "my comments";
  meta.index_writer_options()["write_index_cache"] = true;
  meta.index_reader_options()["read_index_cache"] = true;

  // dimension of the vectors to index
  uint32_t d = 128;
  // size of the database we plan to index
  size_t nb = 200;
  // size of the query vectors we plan to test
  size_t nq = 10;
  // index save path
  std::string index_path = "/tmp/faiss_hnsw_index";

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

  T_LOG(WARNING) << "Generating query vectors...";
  auto query = RandomVectors(nq, d, /*seed=*/1);
  auto query_col = tenann::ArraySeqView{.data = reinterpret_cast<uint8_t*>(query.data()),
                                        .dim = d,
                                        .size = static_cast<uint32_t>(nq),
                                        .elem_type = tenann::PrimitiveType::kFloatType};

  // build and write index
  auto index_builder1 = tenann::IndexFactory::CreateBuilderFromMeta(meta);

  index_builder1->Open(index_path)
      .Add({base_col})
      .Flush();

  meta.search_params()["efSearch"] = 900;
  auto ann_searcher = tenann::AnnSearcherFactory::CreateSearcherFromMeta(meta);

  // load index from disk file
  ann_searcher->ReadIndex(index_path);
  T_DCHECK(ann_searcher->is_index_loaded());

  int k = 10;
  std::vector<int64_t> result_ids(nq * k);
  std::vector<float> result_distances(nq * k);

  // search index
  for (int i = 0; i < nq; i++) {
    auto query_view =
        tenann::PrimitiveSeqView{.data = reinterpret_cast<uint8_t*>(query.data() + i * d),
                                 .size = d,
                                 .elem_type = tenann::PrimitiveType::kFloatType};
    ann_searcher->AnnSearch(query_view, k, result_ids.data() + i * k,
                            reinterpret_cast<uint8_t*>(result_distances.data() + i * k));
  }

  std::cout << "HNSW Results: \n";
  PrintResults(result_ids, result_distances, nq, k);

  // brute force
  tenann::util::BruteForceTopKSearch(d, base_col, nullptr, nullptr, query_col, metric, k,
                              result_ids.data(), result_distances.data());
  std::cout << "Bruteforce Results: \n";
  PrintResults(result_ids, result_distances, nq, k);
}