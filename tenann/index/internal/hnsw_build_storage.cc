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
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>

#include "faiss/impl/FaissAssert.h"
#include "tenann/common/logging.h"

namespace tenann {

namespace {

#ifdef MAP_NORESERVE
constexpr int kMapFlags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE;
#else
constexpr int kMapFlags = MAP_PRIVATE | MAP_ANONYMOUS;
#endif

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
  void* p;
  if (old == nullptr) {
    accountant->Charge(static_cast<int64_t>(new_bytes));
    p = mmap(nullptr, new_bytes, PROT_READ | PROT_WRITE, kMapFlags, -1, 0);
    if (p == MAP_FAILED) accountant->Release(static_cast<int64_t>(new_bytes));
  } else {
#if defined(__linux__)
    accountant->Charge(static_cast<int64_t>(new_bytes - old_bytes));
    p = mremap(old, old_bytes, new_bytes, MREMAP_MAYMOVE);
    if (p == MAP_FAILED) accountant->Release(static_cast<int64_t>(new_bytes - old_bytes));
#else
    // Both blocks are mapped while the rows are copied, so both are charged until the old one goes.
    accountant->Charge(static_cast<int64_t>(new_bytes));
    p = mmap(nullptr, new_bytes, PROT_READ | PROT_WRITE, kMapFlags, -1, 0);
    if (p == MAP_FAILED) {
      accountant->Release(static_cast<int64_t>(new_bytes));
    } else {
      memcpy(p, old, old_bytes);
      munmap(old, old_bytes);
      accountant->Release(static_cast<int64_t>(old_bytes));
    }
#endif
  }
  T_CHECK(p != MAP_FAILED) << "failed to map " << new_bytes << " bytes for HNSW build storage";
  return static_cast<uint8_t*>(p);
}

// Shrinks a block to |new_bytes|, keeping its first |new_bytes|. Returns null for an empty block.
uint8_t* ShrinkBlock(uint8_t* p, size_t old_bytes, size_t new_bytes, MemoryAccountant* accountant) {
  if (p == nullptr) return nullptr;
  if (accountant == nullptr) {
    if (new_bytes == 0) {
      free(p);
      return nullptr;
    }
    void* q = realloc(p, new_bytes);
    return q != nullptr ? static_cast<uint8_t*>(q) : p;  // a failed shrink keeps the larger block
  }
  old_bytes = PageAlign(old_bytes);
  new_bytes = PageAlign(new_bytes);
  if (new_bytes >= old_bytes) return p;
  munmap(p + new_bytes, old_bytes - new_bytes);
  accountant->Release(static_cast<int64_t>(old_bytes - new_bytes));
  return new_bytes == 0 ? nullptr : p;
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

HnswBuildStorage::~HnswBuildStorage() { FreeRows(); }

// The index may be destroyed long after its build (an index cache), on any thread, so a throwing
// accountant must not escape the destructor.
void HnswBuildStorage::FreeRows() noexcept {
  codes = faiss::MaybeOwnedVector<uint8_t>();
  uint8_t* rows = rows_;
  const size_t bytes = capacity_ * code_size;
  rows_ = nullptr;
  capacity_ = 0;
  try {
    FreeBlock(rows, bytes, accountant_.get());
  } catch (const std::exception& e) {
    std::fprintf(stderr, "tenann: releasing HNSW build storage failed: %s\n", e.what());
  } catch (...) {
    std::fprintf(stderr, "tenann: releasing HNSW build storage failed\n");
  }
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
  FreeRows();
  ntotal = 0;
  num_staged_ = 0;
}

void HnswBuildStorage::FinishBuild() {
  T_CHECK_EQ(num_staged_, 0) << "staged rows must be added before the build finishes";
  if (capacity_ == static_cast<size_t>(ntotal)) return;
  rows_ = ShrinkBlock(rows_, capacity_ * code_size, ntotal * code_size, accountant_.get());
  capacity_ = rows_ != nullptr ? ntotal : 0;
  SyncView();
}

size_t HnswBuildStorage::remove_ids(const faiss::IDSelector&) {
  FAISS_THROW_MSG("HnswBuildStorage does not support remove_ids");
}

void HnswBuildStorage::merge_from(faiss::Index&, faiss::idx_t) {
  FAISS_THROW_MSG("HnswBuildStorage does not support merge_from");
}

void HnswBuildStorage::add_sa_codes(faiss::idx_t, const uint8_t*, const faiss::idx_t*) {
  FAISS_THROW_MSG("HnswBuildStorage does not support add_sa_codes");
}

}  // namespace tenann
