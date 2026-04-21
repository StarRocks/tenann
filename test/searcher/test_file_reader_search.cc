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

#include <cstdio>
#include <cstring>
#include <string>

#include "tenann/factory/ann_searcher_factory.h"
#include "tenann/index/default_index_cache.h"
#include "tenann/index/parameters.h"
#include "tenann/store/index_file_reader.h"
#include "test/faiss_test_base.h"

namespace tenann {

/// A local-file-based IndexFileReader that simulates reading from a remote FS.
/// Used for end-to-end testing of the IndexFileReader path.
class TestLocalIndexFileReader : public IndexFileReader {
 public:
  explicit TestLocalIndexFileReader(const std::string& path) : filename_(path), position_(0) {
    fp_ = fopen(path.c_str(), "rb");
    if (fp_) {
      fseek(fp_, 0, SEEK_END);
      file_size_ = ftell(fp_);
      fseek(fp_, 0, SEEK_SET);
    } else {
      file_size_ = 0;
    }
  }

  ~TestLocalIndexFileReader() override {
    if (fp_) fclose(fp_);
  }

  int64_t Read(void* data, int64_t count) override {
    if (!fp_) return -1;
    fseek(fp_, position_, SEEK_SET);
    size_t n = fread(data, 1, count, fp_);
    position_ += n;
    return static_cast<int64_t>(n);
  }

  int64_t ReadAt(int64_t offset, void* data, int64_t count) override {
    if (!fp_) return -1;
    fseek(fp_, offset, SEEK_SET);
    size_t n = fread(data, 1, count, fp_);
    return static_cast<int64_t>(n);
  }

  void Seek(int64_t position) override { position_ = position; }

  int64_t GetSize() override { return file_size_; }

  const std::string& filename() const override { return filename_; }

 private:
  std::string filename_;
  FILE* fp_ = nullptr;
  int64_t position_ = 0;
  int64_t file_size_ = 0;
};

// ============================================================================
// HNSW tests via IndexFileReader
// ============================================================================

class FaissHnswFileReaderSearchTest : public FaissTestBase {
 public:
  FaissHnswFileReaderSearchTest() : FaissTestBase() {
    InitFaissHnswMeta();
    faiss_hnsw_index_builder_ = IndexFactory::CreateBuilderFromMeta(faiss_hnsw_meta_);
  }
};

TEST_F(FaissHnswFileReaderSearchTest, ReadIndex_ViaFileReader) {
  // Build and write HNSW index to disk using the normal path
  CreateAndWriteFaissHnswIndex(false);

  // Now read it back via IndexFileReader
  auto file_reader =
      std::make_shared<TestLocalIndexFileReader>(index_with_primary_key_path());

  auto ann_searcher = AnnSearcherFactory::CreateSearcherFromMeta(meta_);
  ann_searcher->ReadIndex(file_reader);

  EXPECT_TRUE(ann_searcher->is_index_loaded());

  // Search and check recall
  result_ids_.resize(nq_ * k_);
  for (int i = 0; i < nq_; i++) {
    ann_searcher->AnnSearch(query_view_[i], k_, result_ids_.data() + i * k_);
  }
  EXPECT_TRUE(RecallCheckResult_80Percent());
}

TEST_F(FaissHnswFileReaderSearchTest, ReadIndex_ViaFileReader_WithCustomRowId) {
  CreateAndWriteFaissHnswIndex(true);

  auto file_reader =
      std::make_shared<TestLocalIndexFileReader>(index_with_primary_key_path());

  auto ann_searcher = AnnSearcherFactory::CreateSearcherFromMeta(meta_);
  ann_searcher->ReadIndex(file_reader);

  EXPECT_TRUE(ann_searcher->is_index_loaded());

  result_ids_.resize(nq_ * k_);
  for (int i = 0; i < nq_; i++) {
    ann_searcher->AnnSearch(query_view_[i], k_, result_ids_.data() + i * k_);
  }
  EXPECT_TRUE(RecallCheckResult_80Percent());
}

// ============================================================================
// IVFPQ tests via IndexFileReader
// ============================================================================

class FaissIvfPqFileReaderSearchTest : public FaissTestBase {
 public:
  FaissIvfPqFileReaderSearchTest() : FaissTestBase() {
    InitFaissIvfPqMeta();
    faiss_ivf_pq_index_builder_ = IndexFactory::CreateBuilderFromMeta(faiss_ivf_pq_meta_);
  }
};

TEST_F(FaissIvfPqFileReaderSearchTest, ReadIndex_ViaFileReader) {
  // Build IVFPQ index
  CreateAndWriteFaissIvfPqIndex(false);

  // Read via IndexFileReader
  auto file_reader =
      std::make_shared<TestLocalIndexFileReader>(index_with_primary_key_path());

  auto ann_searcher = AnnSearcherFactory::CreateSearcherFromMeta(meta_);
  ann_searcher->ReadIndex(file_reader);

  EXPECT_TRUE(ann_searcher->is_index_loaded());

  result_ids_.resize(nq_ * k_);
  for (int i = 0; i < nq_; i++) {
    ann_searcher->AnnSearch(query_view_[i], k_, result_ids_.data() + i * k_);
  }
  EXPECT_TRUE(RecallCheckResult_80Percent());
}

TEST_F(FaissIvfPqFileReaderSearchTest, ReadIndex_ViaFileReader_WithCustomRowId) {
  CreateAndWriteFaissIvfPqIndex(true);

  auto file_reader =
      std::make_shared<TestLocalIndexFileReader>(index_with_primary_key_path());

  auto ann_searcher = AnnSearcherFactory::CreateSearcherFromMeta(meta_);
  ann_searcher->ReadIndex(file_reader);

  EXPECT_TRUE(ann_searcher->is_index_loaded());

  result_ids_.resize(nq_ * k_);
  for (int i = 0; i < nq_; i++) {
    ann_searcher->AnnSearch(query_view_[i], k_, result_ids_.data() + i * k_);
  }
  EXPECT_TRUE(RecallCheckResult_80Percent());
}

TEST_F(FaissIvfPqFileReaderSearchTest, ReadIndex_ViaFileReader_BlockCache) {
  // Enable block cache mode for IVFPQ
  faiss_ivf_pq_meta_.index_reader_options()["cache_index_block"] = true;

  CreateAndWriteFaissIvfPqIndex(false);

  auto file_reader =
      std::make_shared<TestLocalIndexFileReader>(index_with_primary_key_path());

  auto ann_searcher = AnnSearcherFactory::CreateSearcherFromMeta(meta_);
  // index_cache() returns IndexCache*, which doesn't expose SetCapacity.
  // Drive the global DefaultIndexCache singleton directly (searchers resolve it
  // via GetGlobalIndexCache()).
  DefaultIndexCache::GetGlobalInstance()->SetCapacity(500 * 1024);  // limit 500KB
  ann_searcher->ReadIndex(file_reader);

  EXPECT_TRUE(ann_searcher->is_index_loaded());

  result_ids_.resize(nq_ * k_);
  for (int i = 0; i < nq_; i++) {
    ann_searcher->AnnSearch(query_view_[i], k_, result_ids_.data() + i * k_);
  }
  EXPECT_TRUE(RecallCheckResult_80Percent());
}

TEST_F(FaissIvfPqFileReaderSearchTest, ReadIndex_ViaFileReader_MultiAdd) {
  MultiAddCreateAndWriteFaissIvfPqIndex();

  auto file_reader =
      std::make_shared<TestLocalIndexFileReader>(index_with_primary_key_path());

  auto ann_searcher = AnnSearcherFactory::CreateSearcherFromMeta(meta_);
  ann_searcher->ReadIndex(file_reader);

  EXPECT_TRUE(ann_searcher->is_index_loaded());

  result_ids_.resize(nq_ * k_);
  for (int i = 0; i < nq_; i++) {
    ann_searcher->AnnSearch(query_view_[i], k_, result_ids_.data() + i * k_);
  }
  EXPECT_TRUE(RecallCheckResult_80Percent());
}

}  // namespace tenann
