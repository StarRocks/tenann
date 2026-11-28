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

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "faiss/IndexFlat.h"
#include "faiss/utils/distances.h"
#include "gtest/gtest.h"
#include "tenann/index/default_index_cache.h"
#include "tenann/index/index.h"
#include "tenann/index/index_ivfpq_reader.h"
#include "tenann/index/index_ivfpq_util.h"
#include "tenann/index/internal/index_ivfpq.h"
#include "tenann/store/index_file_reader.h"
#include "tenann/util/random.h"

const float float_diff_threshold = 0.000001;

namespace {

class ConcurrentIndexFileReader : public tenann::IndexFileReader {
 public:
  explicit ConcurrentIndexFileReader(std::string payload, std::chrono::milliseconds delay = {})
      : payload_(std::move(payload)), delay_(delay) {}

  int64_t Read(void* data, int64_t count) override { return -1; }

  int64_t ReadAt(int64_t offset, void* data, int64_t count) override {
    read_count_.fetch_add(1, std::memory_order_relaxed);
    int active = active_reads_.fetch_add(1, std::memory_order_acq_rel) + 1;
    int observed = max_active_reads_.load(std::memory_order_relaxed);
    while (active > observed &&
           !max_active_reads_.compare_exchange_weak(observed, active, std::memory_order_relaxed)) {
    }
    std::this_thread::sleep_for(delay_);

    if (offset < 0 || count < 0 || offset + count > static_cast<int64_t>(payload_.size())) {
      active_reads_.fetch_sub(1, std::memory_order_release);
      return -1;
    }
    memcpy(data, payload_.data() + offset, count);
    active_reads_.fetch_sub(1, std::memory_order_release);
    return count;
  }

  void Seek(int64_t position) override {}
  int64_t GetSize() override { return payload_.size(); }
  const std::string& filename() const override { return filename_; }

  int read_count() const { return read_count_.load(std::memory_order_relaxed); }
  int max_active_reads() const { return max_active_reads_.load(std::memory_order_relaxed); }

 private:
  std::string payload_;
  std::chrono::milliseconds delay_;
  std::string filename_ = "remote-index";
  std::atomic<int> read_count_{0};
  std::atomic<int> active_reads_{0};
  std::atomic<int> max_active_reads_{0};
};

class FailingIndexFileReader : public tenann::IndexFileReader {
 public:
  int64_t Read(void* data, int64_t count) override { return -1; }
  int64_t ReadAt(int64_t offset, void* data, int64_t count) override { return -1; }
  void Seek(int64_t position) override {}
  int64_t GetSize() override { return 4096; }
  const std::string& filename() const override { return filename_; }

 private:
  std::string filename_ = "failing-index";
};

void WaitAndStartThreads(std::vector<std::thread>* threads, std::atomic<int>* ready,
                         std::atomic<bool>* start, int expected_threads) {
  while (ready->load(std::memory_order_acquire) != expected_threads) {
    std::this_thread::yield();
  }
  start->store(true, std::memory_order_release);
  for (auto& thread : *threads) {
    thread.join();
  }
}

}  // namespace

TEST(IndexIvfPqTest, test_reconstruction_error) {
  const int dim = 8;
  const int m = 2;
  const int nlist = 2;
  const int nbits = 8;
  const int nb = 1024;

  auto base = tenann::RandomVectors(nb, dim, 0);
  faiss::IndexFlatL2 coarse_quantizer(dim);
  tenann::IndexIvfPq ivfpq(&coarse_quantizer, dim, nlist, m, nbits);
  ivfpq.train(nb, base.data());
  ivfpq.add(nb, base.data());

  std::vector<float> recons(dim);
  for (int list_no = 0; list_no < nlist; list_no++) {
    auto size = ivfpq.get_list_size(list_no);
    for (auto offset = 0; offset < size; offset++) {
      auto id = ivfpq.invlists->get_single_id(list_no, offset);
      ivfpq.reconstruct_from_offset(list_no, offset, recons.data());
      float actual_squared_error = faiss::fvec_L2sqr(base.data() + dim * id, recons.data(), dim);
      float actual_error = sqrtf(actual_squared_error);
      EXPECT_LE(actual_error - ivfpq.reconstruction_errors[list_no][offset], float_diff_threshold);
    }
  }
}

TEST(IndexIvfPqTest, test_index_ivfpq_util) {
  tenann::IndexMeta meta;
  meta.SetMetaVersion(0);
  meta.SetIndexType(tenann::IndexType::kFaissIvfPq);
  meta.SetIndexFamily(tenann::IndexFamily::kVectorIndex);

  // deafult pq8 requires at least 256 rows
  auto min1 = tenann::GetIvfPqMinRows(meta, 1);
  EXPECT_EQ(min1, 256);

  // ivf-300 requires at least 300 rows
  meta.index_params()["nlist"] = 300;
  auto min2 = tenann::GetIvfPqMinRows(meta, 1);
  EXPECT_EQ(min2, 300);
}

TEST(IndexIvfPqTest, block_cache_single_flights_same_list_miss) {
  constexpr int kThreads = 8;
  tenann::DefaultIndexCache cache(1024 * 1024);
  auto reader = std::make_shared<ConcurrentIndexFileReader>(std::string(64, 'a'),
                                                            std::chrono::milliseconds(10));
  faiss::BlockCacheInvertedLists lists(1, 1, "remote-index", &cache);
  lists.file_reader = reader;
  lists.one_entry_size = 1;
  lists.lists[0].offset = 0;
  lists.lists[0].size = 64;
  lists.cache_keys[0] = "same-list";

  std::atomic<int> ready{0};
  std::atomic<bool> start{false};
  std::vector<uint8_t> results(kThreads);
  std::vector<std::thread> threads;
  for (int i = 0; i < kThreads; ++i) {
    threads.emplace_back([&, i] {
      ready.fetch_add(1, std::memory_order_release);
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      results[i] = lists.get_ptr(0)[0];
    });
  }
  WaitAndStartThreads(&threads, &ready, &start, kThreads);

  EXPECT_EQ(reader->read_count(), 1);
  for (uint8_t result : results) {
    EXPECT_EQ(result, 'a');
  }
}

TEST(IndexIvfPqTest, block_cache_keeps_different_remote_list_reads_concurrent) {
  constexpr int kThreads = 4;
  constexpr size_t kListSize = 64;
  std::string payload(kThreads * kListSize, '\0');
  for (int i = 0; i < kThreads; ++i) {
    std::fill_n(payload.begin() + i * kListSize, kListSize, static_cast<char>('a' + i));
  }

  tenann::DefaultIndexCache cache(1024 * 1024);
  auto reader = std::make_shared<ConcurrentIndexFileReader>(payload, std::chrono::milliseconds(50));
  faiss::BlockCacheInvertedLists lists(kThreads, 1, "remote-index", &cache);
  lists.file_reader = reader;
  lists.one_entry_size = 1;
  for (int i = 0; i < kThreads; ++i) {
    lists.lists[i].offset = i * kListSize;
    lists.lists[i].size = kListSize;
    lists.cache_keys[i] = "different-list-" + std::to_string(i);
  }

  std::atomic<int> ready{0};
  std::atomic<bool> start{false};
  std::vector<int> content_matches(kThreads);
  std::vector<std::thread> threads;
  for (int i = 0; i < kThreads; ++i) {
    threads.emplace_back([&, i] {
      ready.fetch_add(1, std::memory_order_release);
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      const uint8_t* data = lists.get_ptr(i);
      content_matches[i] = memcmp(data, payload.data() + i * kListSize, kListSize) == 0;
    });
  }
  WaitAndStartThreads(&threads, &ready, &start, kThreads);

  EXPECT_EQ(reader->read_count(), kThreads);
  EXPECT_GT(reader->max_active_reads(), 1);
  for (int content_match : content_matches) {
    EXPECT_TRUE(content_match);
  }
}

TEST(IndexIvfPqTest, block_cache_reads_different_local_lists_by_position) {
  constexpr int kThreads = 8;
  constexpr size_t kListSize = 4096;
  char path[] = "/tmp/tenann-block-cache-concurrent-XXXXXX";
  int fd = mkstemp(path);
  ASSERT_NE(fd, -1);
  unlink(path);

  std::string payload(kThreads * kListSize, '\0');
  for (int i = 0; i < kThreads; ++i) {
    std::fill_n(payload.begin() + i * kListSize, kListSize, static_cast<char>('a' + i));
  }
  ASSERT_EQ(pwrite(fd, payload.data(), payload.size(), 0), static_cast<ssize_t>(payload.size()));

  tenann::DefaultIndexCache cache(2 * payload.size());
  faiss::BlockCacheInvertedLists lists(kThreads, 1, path, &cache);
  lists.fd = fd;
  lists.block_size = kListSize;
  lists.totsize = payload.size();
  lists.one_entry_size = 1;
  for (int i = 0; i < kThreads; ++i) {
    lists.lists[i].offset = i * kListSize;
    lists.lists[i].size = kListSize;
    lists.cache_keys[i] = "local-list-" + std::to_string(i);
  }

  std::atomic<int> ready{0};
  std::atomic<bool> start{false};
  std::vector<int> content_matches(kThreads);
  std::vector<std::thread> threads;
  for (int i = 0; i < kThreads; ++i) {
    threads.emplace_back([&, i] {
      ready.fetch_add(1, std::memory_order_release);
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      const uint8_t* data = lists.get_ptr(i);
      content_matches[i] = memcmp(data, payload.data() + i * kListSize, kListSize) == 0;
    });
  }
  WaitAndStartThreads(&threads, &ready, &start, kThreads);

  for (int content_match : content_matches) {
    EXPECT_TRUE(content_match);
  }
}

TEST(IndexIvfPqTest, block_cache_releases_remote_buffer_after_read_failure) {
  tenann::DefaultIndexCache cache(1024 * 1024);
  auto reader = std::make_shared<FailingIndexFileReader>();
  faiss::BlockCacheInvertedLists lists(1, 1, "failing-index", &cache);
  lists.file_reader = reader;
  lists.one_entry_size = 1;
  lists.lists[0].offset = 0;
  lists.lists[0].size = 4096;
  lists.cache_keys[0] = "read-failure";

  // The ownership assertion is provided by LeakSanitizer in the ASAN build.
  for (int i = 0; i < 3; ++i) {
    EXPECT_ANY_THROW(lists.get_ptr(0));
  }
}

TEST(IndexIvfPqTest, block_cache_charges_allocated_local_buffer_size) {
  char path[] = "/tmp/tenann-block-cache-XXXXXX";
  int fd = mkstemp(path);
  ASSERT_NE(fd, -1);
  unlink(path);

  const std::string payload = "0123456789";
  ASSERT_EQ(write(fd, payload.data(), payload.size()), static_cast<ssize_t>(payload.size()));

  tenann::DefaultIndexCache cache(1024 * 1024);
  faiss::BlockCacheInvertedLists lists(1, 1, path, &cache);
  lists.fd = fd;
  lists.block_size = sizeof(void*);
  lists.totsize = payload.size();
  lists.one_entry_size = 1;
  lists.lists[0].offset = 0;
  lists.lists[0].size = payload.size();
  lists.cache_keys[0] = "local-aligned-buffer";

  const uint8_t* data = lists.get_ptr(0);
  EXPECT_EQ(memcmp(data, payload.data(), payload.size()), 0);
  EXPECT_EQ(cache.memory_usage(), 2 * sizeof(void*));
}

TEST(IndexIvfPqTest, block_cache_top_level_charge_excludes_list_payloads) {
  faiss::IndexFlatL2 quantizer(8);
  auto* ivfpq = new tenann::IndexIvfPq(&quantizer, 8, 2, 2, 8);
  auto* lists = new faiss::BlockCacheInvertedLists(2, ivfpq->code_size, "remote-index", nullptr);
  lists->lists[0].size = 1000000;
  lists->lists[1].size = 2000000;
  lists->cache_keys[0] = "metadata-list-0";
  lists->cache_keys[1] = "metadata-list-1";
  ivfpq->replace_invlists(lists, true);

  tenann::Index index(ivfpq, tenann::IndexType::kFaissIvfPq,
                      [](void* raw) { delete static_cast<faiss::Index*>(raw); });
  size_t large_lists_estimate = index.EstimateMemoryUsage();
  lists->lists[0].size = 1;
  lists->lists[1].size = 2;

  EXPECT_EQ(index.EstimateMemoryUsage(), large_lists_estimate);
}

TEST(IndexIvfPqTest, block_cache_get_ids_returns_8byte_aligned_pointers) {
  constexpr size_t kCodeSize = 2;
  constexpr size_t kOneEntrySize = kCodeSize + sizeof(faiss::idx_t);
  constexpr size_t kBlockSize = 64;
  // Leading padding ensures list file offsets are not multiples of 8; list 1
  // also ends 1 byte before a block_size boundary to exercise block-edge padding.
  const std::vector<size_t> list_sizes = {3, 3, 5, 4, 0};
  const std::vector<size_t> list_offsets = {13, 97, 129, 193, 240};

  std::string payload(256, '\x5a');
  std::vector<std::vector<uint8_t>> expected_codes(list_sizes.size());
  std::vector<std::vector<faiss::idx_t>> expected_ids(list_sizes.size());

  for (size_t i = 0; i < list_sizes.size(); ++i) {
    const size_t n = list_sizes[i];
    expected_codes[i].resize(n * kCodeSize);
    expected_ids[i].resize(n);
    for (size_t j = 0; j < expected_codes[i].size(); ++j) {
      expected_codes[i][j] = static_cast<uint8_t>((i * 31 + j * 7 + 1) & 0xff);
    }
    for (size_t j = 0; j < n; ++j) {
      expected_ids[i][j] = static_cast<faiss::idx_t>(100000 * (i + 1) + j * 13 + 5);
    }
    if (n > 0) {
      const size_t codes_bytes = n * kCodeSize;
      const size_t ids_bytes = n * sizeof(faiss::idx_t);
      std::memcpy(payload.data() + list_offsets[i], expected_codes[i].data(), codes_bytes);
      std::memcpy(payload.data() + list_offsets[i] + codes_bytes, expected_ids[i].data(),
                  ids_bytes);
    }
  }

  auto verify_lists = [&](const faiss::BlockCacheInvertedLists& lists) {
    // Check both cache-miss (pass 0) and cache-hit (pass 1) paths.
    for (int pass = 0; pass < 2; ++pass) {
      for (size_t i = 0; i < list_sizes.size(); ++i) {
        const size_t n = list_sizes[i];
        const faiss::idx_t* ids = lists.get_ids(i);
        if (n == 0) {
          EXPECT_EQ(ids, nullptr);
          continue;
        }
        ASSERT_NE(ids, nullptr);
        EXPECT_EQ(reinterpret_cast<uintptr_t>(ids) % alignof(faiss::idx_t), 0u)
            << "list " << i << " pass " << pass;
        for (size_t j = 0; j < n; ++j) {
          EXPECT_EQ(ids[j], expected_ids[i][j]) << "list " << i << " entry " << j;
        }
        const uint8_t* codes = lists.get_codes(i);
        ASSERT_NE(codes, nullptr);
        EXPECT_EQ(std::memcmp(codes, expected_codes[i].data(), expected_codes[i].size()), 0)
            << "list " << i << " pass " << pass;
      }
    }
  };

  // Remote IndexFileReader path.
  {
    tenann::DefaultIndexCache cache(1024 * 1024);
    auto reader = std::make_shared<ConcurrentIndexFileReader>(payload);
    faiss::BlockCacheInvertedLists lists(list_sizes.size(), kCodeSize, "remote-aligned", &cache);
    lists.file_reader = reader;
    lists.totsize = payload.size();
    lists.one_entry_size = kOneEntrySize;
    for (size_t i = 0; i < list_sizes.size(); ++i) {
      lists.lists[i].size = list_sizes[i];
      lists.lists[i].capacity = list_sizes[i];
      lists.lists[i].offset = list_offsets[i];
      lists.offset_difference[i] = 0;
      lists.cache_keys[i] = "remote-aligned-" + std::to_string(i);
    }
    verify_lists(lists);
  }

  // Local POSIX pread path.
  {
    char path[] = "/tmp/tenann-block-cache-align-XXXXXX";
    int fd = mkstemp(path);
    ASSERT_NE(fd, -1);
    unlink(path);
    ASSERT_EQ(pwrite(fd, payload.data(), payload.size(), 0), static_cast<ssize_t>(payload.size()));

    tenann::DefaultIndexCache cache(1024 * 1024);
    faiss::BlockCacheInvertedLists lists(list_sizes.size(), kCodeSize, path, &cache);
    lists.fd = fd;
    lists.block_size = kBlockSize;
    lists.totsize = payload.size();
    lists.one_entry_size = kOneEntrySize;
    for (size_t i = 0; i < list_sizes.size(); ++i) {
      lists.lists[i].size = list_sizes[i];
      lists.lists[i].capacity = list_sizes[i];
      lists.lists[i].offset = list_offsets[i];
      const size_t aligned_offset = (list_offsets[i] / kBlockSize) * kBlockSize;
      lists.offset_difference[i] = list_offsets[i] - aligned_offset;
      lists.cache_keys[i] = "local-aligned-" + std::to_string(i);
    }
    verify_lists(lists);
  }
}
