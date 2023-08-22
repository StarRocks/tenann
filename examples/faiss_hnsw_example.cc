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
  using namespace tenann;
  IndexMeta meta;

  // set meta values
  meta.SetMetaVersion(0);
  meta.SetIndexFamily(IndexFamily::kVectorIndex);
  meta.SetIndexType(IndexType::kFaissHnsw);
  meta.common_params()["dim"] = 128;
  meta.common_params()["is_vector_normed"] = false;
  meta.common_params()["metric_type"] = MetricType::kL2Distance;
  meta.index_params()["efConstruction"] = 40;
  meta.search_params()["efSearch"] = 40;
  meta.extra_params()["comments"] = "my comments";

  // dimension of the vectors to index
  uint32_t d = 128;
  // size of the database we plan to index
  size_t nb = 200 * 1000;
  // size of the query vectors we plan to test
  size_t nq = 1000;
  // index save path
  constexpr const char* index_path = "tmp/tnn_hnsw_index";

  // generate data and query
  LOG(WARNING) << "Generating base vectors...";
  auto base = RandomVectors(nb, d);
  auto base_view = ArraySeqView{
      .data = reinterpret_cast<uint8_t*>(base.data()), .dim = d, .size = static_cast<uint32_t>(nb)};

  LOG(WARNING) << "Generating query vectors...";
  auto query = RandomVectors(nq, d, /*seed=*/1);

  // build and write index
  try {
    auto index_builder = IndexFactory::CreateBuilderFromMeta(meta);
    auto index_writer = IndexFactory::CreateWriterFromMeta(meta);

    index_builder->SetIndexWriter(index_writer.get())
        .Build({base_view})
        .WriteIndex(index_path, false);
  } catch (InternalError& e) {
    std::cerr << "Exception caught: " << e.what() << "\n";
  }

  // read and search index
  try {
    auto index_reader = IndexFactory::CreateReaderFromMeta(meta);
    auto ann_searcher = AnnSearcherFactory::CreateSearcherFromMeta(meta);

    // load index from disk file
    ann_searcher->SetIndexReader(index_reader.get()).ReadIndex(index_path);
    assert(ann_searcher->is_index_loaded());

    int k = 10;
    std::vector<int64_t> result_ids(nq * k);

    // search index
    for (int i = 0; i < nq; i++) {
      auto query_view = PrimitiveSeqView{.data = reinterpret_cast<uint8_t*>(query.data() + i * d),
                                         .size = d,
                                         .elem_type = PrimitiveType::kFloatType};
      ann_searcher->AnnSearch(query_view, k, result_ids.data() + i * k);
    }
  } catch (InternalError& e) {
    std::cerr << "Exception caught: " << e.what() << "\n";
  }
}