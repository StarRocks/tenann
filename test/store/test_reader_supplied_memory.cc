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
#include <vector>

#include "faiss/impl/maybe_owned_vector.h"
#include "tenann/common/error.h"
#include "tenann/factory/ann_searcher_factory.h"
#include "tenann/factory/index_factory.h"
#include "tenann/store/index_file_reader.h"
#include "test/faiss_test_base.h"

namespace tenann {

/// Owns one heap block on behalf of a FAISS view. The block is poisoned before it is
/// freed, so a view left dangling reads 0xDD rather than bytes that happen to survive.
class CountingOwner : public faiss::MaybeOwnedVectorOwner {
 public:
  CountingOwner(uint8_t* p, size_t bytes, int* live) : p_(p), bytes_(bytes), live_(live) {
    ++(*live_);
  }
  ~CountingOwner() override {
    std::memset(p_, 0xDD, bytes_);
    delete[] p_;
    --(*live_);
  }

 private:
  uint8_t* p_;
  size_t bytes_;
  int* live_;
};

/// Serves the index from memory and, depending on how it is configured, supplies the
/// memory FAISS deserialises into. Both modes read from the same staged bytes, so the
/// only variable between arms is who allocated the destination.
class SupplyingIndexFileReader : public IndexFileReader {
 public:
  /// `supply_from` is the 1-based index of the first allocation request to satisfy;
  /// everything before it is declined. 1 supplies all of them, INT_MAX supplies none,
  /// and 2 produces an index half of whose arrays came from us -- the mixed state a
  /// memory limit reached mid-load would produce in production.
  explicit SupplyingIndexFileReader(const std::string& path, int supply_from = 1,
                                    int* live_owners = nullptr, int fail_read_at = -1)
      : filename_(path),
        supply_from_(supply_from),
        live_owners_(live_owners == nullptr ? &own_live_ : live_owners),
        fail_read_at_(fail_read_at) {
    FILE* fp = fopen(path.c_str(), "rb");
    if (fp == nullptr) return;
    fseek(fp, 0, SEEK_END);
    int64_t size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    staged_.resize(size);
    size_ = static_cast<int64_t>(fread(staged_.data(), 1, size, fp));
    fclose(fp);
  }

  int64_t Read(void* data, int64_t count) override {
    if (position_ < 0 || count > size_ - position_) return -1;
    ++read_calls_;
    if (fail_read_at_ > 0 && read_calls_ >= fail_read_at_) return -1;
    std::memcpy(data, staged_.data() + position_, count);
    position_ += count;
    return count;
  }

  int64_t ReadAt(int64_t offset, void* data, int64_t count) override {
    if (offset < 0 || count > size_ - offset) return -1;
    std::memcpy(data, staged_.data() + offset, count);
    return count;
  }

  void Seek(int64_t position) override { position_ = position; }
  int64_t GetSize() override { return size_; }
  const std::string& filename() const override { return filename_; }

  void* AllocateForRead(size_t bytes,
                        std::shared_ptr<faiss::MaybeOwnedVectorOwner>* owner) override {
    ++allocate_calls_;
    allocated_bytes_.push_back(bytes);
    if (allocate_calls_ < supply_from_) return nullptr;
    auto* p = new uint8_t[bytes];
    *owner = std::make_shared<CountingOwner>(p, bytes, live_owners_);
    ++supplied_calls_;
    return p;
  }

  int read_calls() const { return read_calls_; }
  int allocate_calls() const { return allocate_calls_; }
  int supplied_calls() const { return supplied_calls_; }
  const std::vector<size_t>& allocated_bytes() const { return allocated_bytes_; }

 private:
  std::string filename_;
  int supply_from_;
  int* live_owners_;
  int own_live_ = 0;
  int fail_read_at_;
  std::vector<uint8_t> staged_;
  int64_t size_ = 0;
  int64_t position_ = 0;
  int read_calls_ = 0;
  int allocate_calls_ = 0;
  int supplied_calls_ = 0;
  std::vector<size_t> allocated_bytes_;
};

class ReaderSuppliedMemoryTest : public FaissTestBase {
 public:
  ReaderSuppliedMemoryTest() : FaissTestBase() {
    InitFaissHnswMeta();
    faiss_hnsw_index_builder_ = IndexFactory::CreateBuilderFromMeta(faiss_hnsw_meta_);
  }

 protected:
  /// Read through `reader`, bypassing the index cache: the cache is keyed by filename,
  /// so two arms reading the same file would compare an index against itself.
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

/// The default must be invisible: a reader that declines every request has to produce
/// exactly the index FAISS would have built on its own.
TEST_F(ReaderSuppliedMemoryTest, DecliningEveryRequestKeepsThePlainPath) {
  CreateAndWriteFaissHnswIndex(false);

  auto declining = std::make_shared<SupplyingIndexFileReader>(index_with_primary_key_path(),
                                                              /*supply_from=*/INT_MAX);
  auto index = ReadIndexFileWith(declining);
  ASSERT_NE(index, nullptr);

  EXPECT_GT(declining->allocate_calls(), 0) << "the hook should still be consulted";
  EXPECT_EQ(declining->supplied_calls(), 0);

  result_ids_ = SearchWith(index);
  EXPECT_TRUE(RecallCheckResult_80Percent());
}

/// The point of the interface: an index built on supplied memory must be the same index.
/// Compare the ids rather than recall -- a read that landed one field off would still
/// deserialise into something with plausible recall.
TEST_F(ReaderSuppliedMemoryTest, SuppliedMemoryProducesTheSameIndex) {
  CreateAndWriteFaissHnswIndex(false);

  auto plain = std::make_shared<SupplyingIndexFileReader>(index_with_primary_key_path(),
                                                          /*supply_from=*/INT_MAX);
  auto supplied = std::make_shared<SupplyingIndexFileReader>(index_with_primary_key_path(),
                                                             /*supply_from=*/1);
  auto from_plain = SearchWith(ReadIndexFileWith(plain));
  auto from_supplied = SearchWith(ReadIndexFileWith(supplied));

  // Without this the two arms could both be on the plain path and prove nothing.
  EXPECT_GT(supplied->supplied_calls(), 0);

  ASSERT_EQ(from_plain.size(), from_supplied.size());
  EXPECT_EQ(from_plain, from_supplied);
}

/// The dangerous half of the interface. Once the size has been read off the stream the
/// branch cannot hand the read back, so a decision to decline has to finish the read
/// itself. Declining partway through is what a memory limit reached mid-load looks like;
/// if the read pointer slips by even one field, every later field is garbage.
TEST_F(ReaderSuppliedMemoryTest, DecliningPartwayLeavesTheReadPointerIntact) {
  CreateAndWriteFaissHnswIndex(false);

  auto plain = std::make_shared<SupplyingIndexFileReader>(index_with_primary_key_path(),
                                                          /*supply_from=*/INT_MAX);
  auto mixed = std::make_shared<SupplyingIndexFileReader>(index_with_primary_key_path(),
                                                          /*supply_from=*/2);
  auto from_plain = SearchWith(ReadIndexFileWith(plain));
  auto from_mixed = SearchWith(ReadIndexFileWith(mixed));

  EXPECT_GT(mixed->allocate_calls(), mixed->supplied_calls())
      << "this arm has to exercise both outcomes to be worth anything";
  EXPECT_GT(mixed->supplied_calls(), 0);

  ASSERT_EQ(from_plain.size(), from_mixed.size());
  EXPECT_EQ(from_plain, from_mixed);
}

/// The index views the supplied blocks, so it has to keep them alive by itself. Drop the
/// reader that handed them over and search again.
TEST_F(ReaderSuppliedMemoryTest, SuppliedBlocksOutliveTheReader) {
  CreateAndWriteFaissHnswIndex(false);

  int live = 0;
  IndexRef index;
  {
    auto reader = std::make_shared<SupplyingIndexFileReader>(index_with_primary_key_path(),
                                                             /*supply_from=*/1, &live);
    index = ReadIndexFileWith(reader);
    ASSERT_GT(reader->supplied_calls(), 0);
  }
  ASSERT_NE(index, nullptr);
  EXPECT_GT(live, 0) << "the index views those blocks, so it has to own them";

  result_ids_ = SearchWith(index);
  EXPECT_TRUE(RecallCheckResult_80Percent());

  index.reset();
  EXPECT_EQ(live, 0) << "and it has to release them, or every cached index leaks";
}

/// A read that throws after the block was handed out must not leak it. The owner is a
/// local until the view is built, so unwinding releases it.
TEST_F(ReaderSuppliedMemoryTest, AFailedReadReleasesTheSuppliedBlock) {
  CreateAndWriteFaissHnswIndex(false);

  int live = 0;
  {
    // Fail late enough to be past the header and into the large arrays.
    auto reader = std::make_shared<SupplyingIndexFileReader>(
        index_with_primary_key_path(), /*supply_from=*/1, &live, /*fail_read_at=*/3);
    EXPECT_THROW(ReadIndexFileWith(reader), Error);
  }
  EXPECT_EQ(live, 0) << "a block handed out for a read that failed has to come back";
}

/// Without an explicit byte count the cache falls back to the type-based heuristic, which
/// reads `ntotal` and `neighbors.size()` -- both correct for a view. The supplied-memory
/// path therefore needs no explicit size, unlike whole-file staging.
TEST_F(ReaderSuppliedMemoryTest, TheHeuristicSeesThroughTheViews) {
  CreateAndWriteFaissHnswIndex(false);

  auto plain = std::make_shared<SupplyingIndexFileReader>(index_with_primary_key_path(),
                                                          /*supply_from=*/INT_MAX);
  auto supplied = std::make_shared<SupplyingIndexFileReader>(index_with_primary_key_path(),
                                                             /*supply_from=*/1);
  const size_t plain_bytes = ReadIndexFileWith(plain)->EstimateMemoryUsage();
  const size_t supplied_bytes = ReadIndexFileWith(supplied)->EstimateMemoryUsage();

  EXPECT_GT(plain_bytes, 0u);
  EXPECT_EQ(plain_bytes, supplied_bytes)
      << "a viewed index must be charged what an owning one is charged";
}

/// IVF needs its meta built in the constructor, the way every other IVF suite does it:
/// switching meta mid-test leaves the base vectors sized for the HNSW fixture and PQ
/// training fails before any of this code is reached.
class ReaderSuppliedMemoryIvfTest : public FaissTestBase {
 public:
  ReaderSuppliedMemoryIvfTest() : FaissTestBase() {
    InitFaissIvfPqMeta();
    faiss_ivf_pq_index_builder_ = IndexFactory::CreateBuilderFromMeta(faiss_ivf_pq_meta_);
  }

 protected:
  std::vector<int64_t> ReadAndSearch(int supply_from, int* allocate_calls) {
    auto reader = std::make_shared<SupplyingIndexFileReader>(index_with_primary_key_path(),
                                                             supply_from);
    auto index_reader = IndexFactory::CreateReaderFromMeta(faiss_ivf_pq_meta_);
    index_reader->SetFileReader(reader);
    auto searcher = AnnSearcherFactory::CreateSearcherFromMeta(faiss_ivf_pq_meta_);
    searcher->AttachIndexRef(index_reader->ReadIndexFile(index_with_primary_key_path()));
    std::vector<int64_t> ids(nq_ * k_);
    for (int i = 0; i < nq_; i++) {
      searcher->AnnSearch(query_view_[i], k_, ids.data() + i * k_);
    }
    if (allocate_calls != nullptr) *allocate_calls = reader->allocate_calls();
    return ids;
  }
};

/// IVF inverted lists are read through read_vector_with_known_size, whose target the
/// caller has already resized. Intercepting there would allocate a second block and throw
/// the first away, so that path must never reach the hook.
TEST_F(ReaderSuppliedMemoryIvfTest, InvertedListsAreNeverIntercepted) {
  CreateAndWriteFaissIvfPqIndex(false);

  int allocate_calls = 0;
  ReadAndSearch(/*supply_from=*/1, &allocate_calls);

  // One request per inverted list is what interception would look like, and there are
  // nlist of them. The coarse quantizer and the codebook are legitimately offered.
  EXPECT_LT(allocate_calls, 16)
      << "requests scale with the inverted lists, so the known-size path was intercepted";
}

/// The whole patch, checked end to end on a second index type: the same file read with
/// and without the hook has to search identically.
TEST_F(ReaderSuppliedMemoryIvfTest, RoundTripsIdentically) {
  CreateAndWriteFaissIvfPqIndex(false);

  EXPECT_EQ(ReadAndSearch(INT_MAX, nullptr), ReadAndSearch(1, nullptr));
}

}  // namespace tenann
