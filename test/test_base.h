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

#include <sys/time.h>

#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "tenann/common/error.h"
#include "tenann/factory/index_factory.h"
#include "tenann/factory/ann_searcher_factory.h"
#include "tenann/builder/faiss_hnsw_index_builder.h"
#include "tenann/searcher/faiss_hnsw_ann_searcher.h"

namespace tenann {

class TestBase : public ::testing::Test {
 public:
  uint32_t& d() { return d_; }
  size_t& nb() { return nb_; }
  size_t& nq() { return nq_; }
  uint32_t& k() { return k_; }
  const char* index_path() { return index_path_; }
  const char* index_with_primary_key_path() { return index_with_primary_key_path_; }

  std::vector<int64_t>& ids() { return ids_; }
  std::vector<float>& base() { return base_; }
  std::vector<uint32_t>& offsets() { return offsets_; }
  std::vector<float>& query_data() { return query_; }
  std::vector<int64_t>& result_ids() { return result_ids_; }
  std::vector<int64_t>& accurate_query_result_ids() { return accurate_query_result_ids_; }

  IndexMeta& meta() { return meta_; }
  PrimitiveSeqView& id_view() { return id_view_; }
  ArraySeqView& base_view() { return base_view_; }
  VlArraySeqView& base_vl_view() { return base_vl_view_; }
  std::unique_ptr<IndexBuilder>& index_builder() { return faiss_hnsw_index_builder_; }
  IndexWriterRef& index_writer() { return index_writer_; }
  IndexReaderRef& index_reader() { return index_reader_; }
  std::unique_ptr<AnnSearcher>& ann_searcher() { return ann_searcher_; }

 protected:
  void SetUp() override;

  void TearDown() override {}

  std::vector<float> RandomVectors(uint32_t n, uint32_t dim, int seed = 0);
  float EuclideanDistance(const float* v1, const float* v2);
  void InitAccurateQueryResult();
  void CreateAndWriteIndex();
  void ReadIndexAndDefaultSearch();

  // log output: build_Release/Testing/Temporary/LastTest.log
  bool CheckResult();

 private:
   // dimension of the vectors to index
  uint32_t d_ = 128;
  // size of the database we plan to index
  size_t nb_ = 200;
  // size of the query vectors we plan to test
  size_t nq_ = 10;
  // top k
  uint32_t k_ = 10;
  // index save path
  const char* index_path_ = "/tmp/faiss_hnsw_index";
  const char* index_with_primary_key_path_ = "/tmp/faiss_hnsw_index_with_ids";

  std::vector<int64_t> ids_;
  std::vector<float> base_;
  std::vector<uint32_t> offsets_;
  std::vector<float> query_;
  std::vector<int64_t> result_ids_;
  std::vector<int64_t> accurate_query_result_ids_;

  IndexMeta meta_;
  PrimitiveSeqView id_view_;
  ArraySeqView base_view_;
  VlArraySeqView base_vl_view_;
  std::unique_ptr<IndexBuilder> faiss_hnsw_index_builder_;
  IndexWriterRef index_writer_;
  IndexReaderRef index_reader_;
  std::unique_ptr<AnnSearcher> ann_searcher_;
};

}  // namespace tenann