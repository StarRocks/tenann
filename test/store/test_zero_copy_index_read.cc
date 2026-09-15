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
#include <memory>
#include <string>

#include "tenann/common/error.h"
#include "tenann/factory/ann_searcher_factory.h"
#include "tenann/factory/index_factory.h"
#include "tenann/store/index_file_reader.h"
#include "test/faiss_test_base.h"

namespace tenann {

/// A reader that never offers a buffer, so it exercises the default hand-off.
class StreamOnlyIndexFileReader : public IndexFileReader {
 public:
  int64_t Read(void*, int64_t) override { return -1; }
  int64_t ReadAt(int64_t, void*, int64_t) override { return -1; }
  void Seek(int64_t) override {}
  int64_t GetSize() override { return 0; }
  const std::string& filename() const override { return filename_; }

 private:
  std::string filename_ = "stream-only";
};

/// Stages the whole file in memory, the way a remote-filesystem reader does, and then
/// either offers those bytes for a zero-copy read or keeps them to itself. Both modes
/// serve Read() from the same buffer, so the only variable between them is whether FAISS
/// views the bytes or copies them.
class StagingIndexFileReader : public IndexFileReader {
 public:
  /// `freed` is set when the staged bytes are released, and they are poisoned first: a
  /// view left dangling then reads 0xDD instead of bytes that happen to still be there.
  /// `truncate_by` shortens the offered buffer without shortening the file, which is what
  /// a reader that mis-sized its staging area would do.
  StagingIndexFileReader(const std::string& path, bool offer_zero_copy, bool* freed = nullptr,
                         int64_t truncate_by = 0)
      : filename_(path), offer_zero_copy_(offer_zero_copy), truncate_by_(truncate_by) {
    FILE* fp = fopen(path.c_str(), "rb");
    if (fp == nullptr) return;
    fseek(fp, 0, SEEK_END);
    int64_t size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    auto* raw = new uint8_t[size];
    size_ = static_cast<int64_t>(fread(raw, 1, size, fp));
    fclose(fp);
    int64_t poison_size = size;
    staged_.reset(raw, [freed, poison_size](uint8_t* p) {
      std::memset(p, 0xDD, poison_size);
      delete[] p;
      if (freed != nullptr) *freed = true;
    });
  }

  int64_t Read(void* data, int64_t count) override {
    if (staged_ == nullptr || position_ < 0 || count > size_ - position_) return -1;
    std::memcpy(data, staged_.get() + position_, count);
    position_ += count;
    read_calls_++;
    return count;
  }

  int64_t ReadAt(int64_t offset, void* data, int64_t count) override {
    if (staged_ == nullptr || offset < 0 || count > size_ - offset) return -1;
    std::memcpy(data, staged_.get() + offset, count);
    return count;
  }

  void Seek(int64_t position) override { position_ = position; }

  int64_t GetSize() override { return size_; }

  const std::string& filename() const override { return filename_; }

  ZeroCopyBuffer TryGetZeroCopyBuffer() override {
    if (!offer_zero_copy_ || staged_ == nullptr) return {};
    return {staged_.get(), size_ - truncate_by_, staged_};
  }

  /// How many times FAISS asked for bytes. Zero means it viewed them instead.
  int read_calls() const { return read_calls_; }

 private:
  std::string filename_;
  bool offer_zero_copy_;
  int64_t truncate_by_ = 0;
  std::shared_ptr<uint8_t> staged_;
  int64_t size_ = 0;
  int64_t position_ = 0;
  int read_calls_ = 0;
};

class ZeroCopyIndexReadTest : public FaissTestBase {
 public:
  ZeroCopyIndexReadTest() : FaissTestBase() {
    InitFaissHnswMeta();
    faiss_hnsw_index_builder_ = IndexFactory::CreateBuilderFromMeta(faiss_hnsw_meta_);
  }

 protected:
  /// Read the index file through `reader`, bypassing the index cache. Searcher::ReadIndex
  /// goes through the cache, which is keyed by filename -- two arms reading the same file
  /// would compare one index against itself and pass no matter what the code does.
  IndexRef ReadIndexFileWith(const IndexFileReaderPtr& reader) {
    auto index_reader = IndexFactory::CreateReaderFromMeta(meta_);
    index_reader->SetFileReader(reader);
    return index_reader->ReadIndexFile(index_with_primary_key_path());
  }

  std::vector<int64_t> SearchWith(IndexRef index) {
    auto searcher = AnnSearcherFactory::CreateSearcherFromMeta(meta_);
    searcher->AttachIndexRef(std::move(index));
    std::vector<int64_t> ids(nq_ * k_);
    for (int i = 0; i < nq_; i++) {
      searcher->AnnSearch(query_view_[i], k_, ids.data() + i * k_);
    }
    return ids;
  }
};

/// Readers that stream keep the default and offer nothing, so nothing about them changes.
TEST_F(ZeroCopyIndexReadTest, AStreamingReaderOffersNothing) {
  StreamOnlyIndexFileReader reader;
  auto buffer = reader.TryGetZeroCopyBuffer();
  EXPECT_EQ(buffer.data, nullptr);
  EXPECT_EQ(buffer.size, 0);
  EXPECT_EQ(buffer.owner, nullptr);
}

/// The point of the interface: viewing the bytes must reconstruct the same index as
/// copying them. Compare the ids themselves rather than recall -- a read that landed on
/// the wrong offset would still deserialise into an index with plausible-looking recall.
TEST_F(ZeroCopyIndexReadTest, ZeroCopyReturnsWhatTheCopyingReadReturns) {
  CreateAndWriteFaissHnswIndex(false);

  auto copying = std::make_shared<StagingIndexFileReader>(index_with_primary_key_path(),
                                                          /*offer_zero_copy=*/false);
  auto viewing = std::make_shared<StagingIndexFileReader>(index_with_primary_key_path(),
                                                          /*offer_zero_copy=*/true);
  auto copied = SearchWith(ReadIndexFileWith(copying));
  auto viewed = SearchWith(ReadIndexFileWith(viewing));

  // Viewing the bytes means FAISS never asks for them. Without this the two arms could
  // both be taking the copying path and the comparison below would prove nothing.
  EXPECT_GT(copying->read_calls(), 0);
  EXPECT_EQ(viewing->read_calls(), 0);

  ASSERT_EQ(copied.size(), viewed.size());
  EXPECT_EQ(copied, viewed);

  result_ids_ = viewed;
  EXPECT_TRUE(RecallCheckResult_80Percent());
}

/// The index holds views into the staged bytes, so it has to keep them alive by itself.
/// Drop the reader and the index reader and search again: if the deleter did not capture
/// the owner token, the bytes are poisoned and freed underneath the index.
TEST_F(ZeroCopyIndexReadTest, StagedBytesOutliveTheReaderThatStagedThem) {
  CreateAndWriteFaissHnswIndex(false);

  bool freed = false;
  IndexRef index;
  {
    auto reader = std::make_shared<StagingIndexFileReader>(index_with_primary_key_path(),
                                                           /*offer_zero_copy=*/true, &freed);
    index = ReadIndexFileWith(reader);
  }
  ASSERT_NE(index, nullptr);
  EXPECT_FALSE(freed) << "the index views those bytes, so it has to own them";

  result_ids_ = SearchWith(index);
  EXPECT_TRUE(RecallCheckResult_80Percent());

  index.reset();
  EXPECT_TRUE(freed) << "and it has to release them, or the buffer leaks";
}

/// With views instead of copies the type-based heuristic cannot see the real footprint,
/// so the buffer size is reported explicitly -- otherwise a cache would over-commit.
TEST_F(ZeroCopyIndexReadTest, TheIndexReportsTheBufferItViews) {
  CreateAndWriteFaissHnswIndex(false);

  auto reader = std::make_shared<StagingIndexFileReader>(index_with_primary_key_path(),
                                                         /*offer_zero_copy=*/true);
  const int64_t file_size = reader->GetSize();
  ASSERT_GT(file_size, 0);

  auto index = ReadIndexFileWith(reader);
  ASSERT_NE(index, nullptr);
  EXPECT_EQ(index->EstimateMemoryUsage(), static_cast<size_t>(file_size));
}

/// A buffer that does not cover the whole file breaks the precondition. FAISS checks the
/// length of every view it takes, so a short buffer has to fail loudly rather than build
/// an index quietly backed by short arrays.
TEST_F(ZeroCopyIndexReadTest, ATruncatedBufferIsRejected) {
  CreateAndWriteFaissHnswIndex(false);

  auto reader = std::make_shared<StagingIndexFileReader>(
      index_with_primary_key_path(), /*offer_zero_copy=*/true, /*freed=*/nullptr,
      /*truncate_by=*/4096);
  EXPECT_THROW(ReadIndexFileWith(reader), Error);
}

}  // namespace tenann
