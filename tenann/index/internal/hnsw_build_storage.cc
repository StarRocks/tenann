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

#include "tenann/index/internal/hnsw_build_storage.h"

#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>

#include "tenann/common/logging.h"

namespace tenann {

namespace {

size_t PageAlign(size_t bytes) {
  static const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  return (bytes + page - 1) / page * page;
}

// Grows a block to |new_bytes|, keeping its contents. With an accountant it is a mapping charged
// to it: pages are committed only when touched, and growth moves page tables instead of copying.
uint8_t* GrowBlock(uint8_t* old, size_t old_bytes, size_t new_bytes, MemoryAccountant* accountant) {
  if (accountant == nullptr) {
    void* p = realloc(old, new_bytes);
    T_CHECK(p != nullptr) << "failed to allocate " << new_bytes << " bytes for HNSW build storage";
    return static_cast<uint8_t*>(p);
  }
  old_bytes = PageAlign(old_bytes);
  new_bytes = PageAlign(new_bytes);
  accountant->Charge(static_cast<int64_t>(new_bytes - old_bytes));
  void* p;
  if (old == nullptr) {
    p = mmap(nullptr, new_bytes, PROT_READ | PROT_WRITE,
             MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  } else {
#if defined(__linux__)
    p = mremap(old, old_bytes, new_bytes, MREMAP_MAYMOVE);
#else
    p = mmap(nullptr, new_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p != MAP_FAILED) {
      memcpy(p, old, old_bytes);
      munmap(old, old_bytes);
    }
#endif
  }
  if (p == MAP_FAILED) {
    accountant->Release(static_cast<int64_t>(new_bytes - old_bytes));
    T_CHECK(false) << "failed to map " << new_bytes << " bytes for HNSW build storage";
  }
  return static_cast<uint8_t*>(p);
}

void FreeBlock(uint8_t* p, size_t bytes, MemoryAccountant* accountant) {
  if (p == nullptr) return;
  if (accountant == nullptr) {
    free(p);
    return;
  }
  munmap(p, PageAlign(bytes));
  accountant->Release(static_cast<int64_t>(PageAlign(bytes)));
}

}  // namespace

HnswBuildStorage::HnswBuildStorage(faiss::idx_t d, faiss::MetricType metric,
                                   MemoryAccountantPtr accountant)
    : faiss::IndexFlat(d, metric), accountant_(std::move(accountant)) {}

HnswBuildStorage::~HnswBuildStorage() {
  codes = faiss::MaybeOwnedVector<uint8_t>();
  FreeBlock(rows_, capacity_ * code_size, accountant_.get());
}

void HnswBuildStorage::Reserve(size_t rows) {
  if (rows <= capacity_) return;
  // A mapping grows without copying, so it grows in small steps and is charged little beyond what
  // it holds; the heap may copy on growth, so it doubles.
  const size_t step = std::max(accountant_ ? capacity_ / 8 : capacity_, static_cast<size_t>(4096));
  const size_t cap = std::max(rows, capacity_ + step);
  rows_ = GrowBlock(rows_, capacity_ * code_size, cap * code_size, accountant_.get());
  capacity_ = cap;
  SyncView();
}

void HnswBuildStorage::SyncView() {
  codes = faiss::MaybeOwnedVector<uint8_t>::create_view(rows_, ntotal * code_size, nullptr);
}

float* HnswBuildStorage::StageRow() {
  Reserve(ntotal + num_staged_ + 1);
  return xb() + (ntotal + num_staged_++) * d;
}

void HnswBuildStorage::add(faiss::idx_t n, const float* x) {
  if (n == 0) return;
  if (x == staged_data()) {
    T_CHECK_LE(n, num_staged_);
    num_staged_ -= n;
  } else {
    T_CHECK_EQ(num_staged_, 0) << "rows can't be added while other rows are staged";
    Reserve(ntotal + n);
    memcpy(xb() + ntotal * d, x, n * code_size);
  }
  ntotal += n;
  SyncView();
}

void HnswBuildStorage::reset() {
  codes = faiss::MaybeOwnedVector<uint8_t>();
  FreeBlock(rows_, capacity_ * code_size, accountant_.get());
  rows_ = nullptr;
  capacity_ = 0;
  ntotal = 0;
  num_staged_ = 0;
}

}  // namespace tenann
