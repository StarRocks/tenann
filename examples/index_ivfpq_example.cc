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

#include "faiss/IndexIVFPQ.h"
#include "tenann/common/logging.h"
#include "tenann/factory/ann_searcher_factory.h"
#include "tenann/factory/index_factory.h"
#include "tenann/store/index_meta.h"

std::vector<float> RandomVectors(uint32_t n, uint32_t dim, int seed = 0) {
  std::mt19937 rng(seed);
  std::vector<float> data(n * dim);
  std::uniform_real_distribution<> distrib;
  for (int64_t i = 0; i < n * dim; i++) {
    data[i] = distrib(rng);
  }
  return data;
}

// 计算向量之间的欧氏距离
float calculateEuclideanDistance(std::vector<float>::const_iterator vec1_begin,
                                 std::vector<float>::const_iterator vec1_end,
                                 std::vector<float>::const_iterator vec2_begin) {
  float distance = 0.0;
  while (vec1_begin != vec1_end) {
    distance += std::pow(*vec1_begin - *vec2_begin, 2);
    vec1_begin++;
    vec2_begin++;
  }
  return std::sqrt(distance);
}

float ComputeRecall(const std::vector<int64_t>& accurate_query_result_ids,
                    const std::vector<int64_t>& result_ids, uint32_t d, int64_t nb, int64_t nq,
                    int64_t k) {
  float recall_sum = 0.0;
  for (int64_t i = 0; i < nq; i++) {
    printf("accurate_id(%zu): \n", i);
    std::set<int64_t> accurate_set;
    for (int64_t j = 0; j < k; j++) {
      auto accurate_id = accurate_query_result_ids[i * k + j];
      accurate_set.insert(accurate_id);
      printf("%d ", accurate_id);
    }
    int hit = 0;

    printf("\nresult_ids(%zu): \n", i);
    for (int64_t j = 0; j < k; j++) {
      auto result_id = result_ids[i * k + j];
      printf("%d ", result_id);
      if (accurate_set.find(result_id) != accurate_set.end()) {
        hit++;
      }
    }
    float recall_i = static_cast<float>(hit) / k;
    printf("query %zu: recall rate:%f\n", i, recall_i);
    recall_sum += recall_i;
  }
  float result = recall_sum / nq;
  printf("Aggregate Recall:%f\n", result);
  return result;
}

// 初始化准确查询结果
std::vector<int64_t> initAccurateQueryResult(const std::vector<float>& base,
                                             const std::vector<float>& query, uint32_t d,
                                             int64_t nb, int64_t nq, int64_t k) {
  std::vector<int64_t> accurate_query_result_ids;
  // 搜索索引
  for (int64_t i = 0; i < nq; i++) {
    std::vector<std::pair<float, int64_t>> distances;
    for (int64_t j = 0; j < nb; j++) {
      float distance = calculateEuclideanDistance(
          query.begin() + i * d, query.begin() + (i + 1) * d, base.begin() + j * d);
      distances.push_back(std::make_pair(distance, j));
    }

    std::sort(distances.begin(), distances.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    for (int64_t j = 0; j < k; j++) {
      accurate_query_result_ids.push_back(distances[j].second);
    }
  }
  return accurate_query_result_ids;
}

int main() {
  tenann::IndexMeta meta;

  // set meta values
  meta.SetMetaVersion(0);
  meta.SetIndexFamily(tenann::IndexFamily::kVectorIndex);
  meta.SetIndexType(tenann::IndexType::kFaissIvfPq);
  meta.common_params()["dim"] = 16;
  meta.common_params()["is_vector_normed"] = true;
  meta.common_params()["metric_type"] =
      tenann::MetricType::kInnerProduct;  // kL2Distance kCosineSimilarity kInnerProduct
  meta.index_params()["M"] = 8;
  meta.search_params()["nbits"] = 8;
  meta.extra_params()["comments"] = "my comments";

  // dimension of the vectors to index
  uint32_t d = meta.common_params()["dim"];
  // size of the database we plan to index
  int64_t nb = 10000;
  // size of the query vectors we plan to test
  int64_t nq = 10;
  int64_t k = 10;
  // index save path
  std::string index_path = "/tmp/tenann_ivfpq_index";

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

  // 执行暴力检索
  std::vector<int64_t> searchResults = initAccurateQueryResult(base, query, d, nb, nq, k);
  // 输出结果集
  std::cout << "Search Results:" << std::endl;
  for (const auto& result : searchResults) {
    std::cout << result << " ";
  }
  std::cout << std::endl;

  // build and write index
  try {
    auto index_builder1 = tenann::IndexFactory::CreateBuilderFromMeta(meta);
    tenann::IndexWriterRef index_writer = tenann::IndexFactory::CreateWriterFromMeta(meta);
    index_builder1->SetIndexWriter(index_writer)
        .SetIndexCache(tenann::IndexCache::GetGlobalInstance())
        .Open(index_path);
    const int64_t step = 1000;
    auto* data = base.data();
    for (int64_t i = 0; i < nb; i += step) {
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
    index_builder1->Close();

    std::shared_ptr<tenann::IndexReader> reader = tenann::IndexFactory::CreateReaderFromMeta(meta);
    auto ann_searcher = tenann::AnnSearcherFactory::CreateSearcherFromMeta(meta);
    ann_searcher
        ->SetIndexReader(reader)  //
        .SetIndexCache(tenann::IndexCache::GetGlobalInstance())
        .ReadIndex(index_path);

    std::vector<int64_t> result_ids(nq * k);

    // search index
    for (int i = 0; i < nq; i++) {
      auto query_view =
          tenann::PrimitiveSeqView{.data = reinterpret_cast<uint8_t*>(query.data() + i * d),
                                   .size = d,
                                   .elem_type = tenann::PrimitiveType::kFloatType};

      ann_searcher->AnnSearch(query_view, k, result_ids.data() + i * k);

      std::cout << "Result of query " << i << ": ";
      for (int j = 0; j < k; j++) {
        std::cout << result_ids[i * k + j] << ",";
      }
      std::cout << "\n";
    }
    std::cout << "召回率： " << ComputeRecall(searchResults, result_ids, d, nb, nq, k);
  } catch (tenann::Error& e) {
    std::cerr << "Exception caught: " << e.what() << "\n";
  }
}