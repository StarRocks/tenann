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
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "faiss/impl/DistanceComputer.h"
#include "faiss/impl/FaissAssert.h"
#include "tenann/common/logging.h"

#if defined(__x86_64__)
#include <immintrin.h>
#endif

namespace tenann {

namespace {

using L2U8Fn = int64_t (*)(const uint8_t*, const uint8_t*, size_t);
using L2U8x4Fn = void (*)(const uint8_t*, const uint8_t* const*, size_t, int64_t*);

int64_t L2U8Scalar(const uint8_t* a, const uint8_t* b, size_t d) {
  int64_t r = 0;
  for (size_t i = 0; i < d; i++) {
    int t = int(a[i]) - int(b[i]);
    r += t * t;
  }
  return r;
}

void L2U8x4Scalar(const uint8_t* q, const uint8_t* const* y, size_t d, int64_t* out) {
  for (int k = 0; k < 4; k++) out[k] = L2U8Scalar(q, y[k], d);
}

#if defined(__x86_64__)
// int32 lanes stay exact: a lane gains at most 2 * 255^2 per step, and each kernel folds its lanes
// into int64 every kFlushDims dimensions.
constexpr size_t kFlushDims = 8192;

__attribute__((target("avx2"))) inline int64_t HsumAvx2(__m256i v) {
  __m256i lo = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(v));
  __m256i hi = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(v, 1));
  __m256i s = _mm256_add_epi64(lo, hi);
  __m128i s2 = _mm_add_epi64(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1));
  return _mm_cvtsi128_si64(s2) + _mm_extract_epi64(s2, 1);
}

__attribute__((target("avx2"))) inline __m256i SqDiffAvx2(__m128i a, __m128i b) {
  __m256i x = _mm256_sub_epi16(_mm256_cvtepu8_epi16(a), _mm256_cvtepu8_epi16(b));
  return _mm256_madd_epi16(x, x);
}

__attribute__((target("avx2"))) int64_t L2U8Avx2(const uint8_t* a, const uint8_t* b, size_t d) {
  int64_t r = 0;
  size_t i = 0;
  while (i + 16 <= d) {
    __m256i acc = _mm256_setzero_si256();
    size_t end = std::min(d - d % 16, i + kFlushDims);
    for (; i < end; i += 16) {
      acc = _mm256_add_epi32(acc, SqDiffAvx2(_mm_loadu_si128((const __m128i*)(a + i)),
                                             _mm_loadu_si128((const __m128i*)(b + i))));
    }
    r += HsumAvx2(acc);
  }
  return r + L2U8Scalar(a + i, b + i, d - i);
}

// Four candidates against one query in a single pass: the four rows' loads are issued together, so
// more of the (random, latency-bound) reads are in flight at once.
__attribute__((target("avx2"))) void L2U8x4Avx2(const uint8_t* q, const uint8_t* const* y, size_t d,
                                                int64_t* out) {
  for (int k = 0; k < 4; k++) out[k] = 0;
  size_t i = 0;
  while (i + 16 <= d) {
    __m256i a0 = _mm256_setzero_si256(), a1 = _mm256_setzero_si256();
    __m256i a2 = _mm256_setzero_si256(), a3 = _mm256_setzero_si256();
    size_t end = std::min(d - d % 16, i + kFlushDims);
    for (; i < end; i += 16) {
      __m128i vq = _mm_loadu_si128((const __m128i*)(q + i));
      a0 = _mm256_add_epi32(a0, SqDiffAvx2(vq, _mm_loadu_si128((const __m128i*)(y[0] + i))));
      a1 = _mm256_add_epi32(a1, SqDiffAvx2(vq, _mm_loadu_si128((const __m128i*)(y[1] + i))));
      a2 = _mm256_add_epi32(a2, SqDiffAvx2(vq, _mm_loadu_si128((const __m128i*)(y[2] + i))));
      a3 = _mm256_add_epi32(a3, SqDiffAvx2(vq, _mm_loadu_si128((const __m128i*)(y[3] + i))));
    }
    out[0] += HsumAvx2(a0);
    out[1] += HsumAvx2(a1);
    out[2] += HsumAvx2(a2);
    out[3] += HsumAvx2(a3);
  }
  for (int k = 0; k < 4; k++) out[k] += L2U8Scalar(q + i, y[k] + i, d - i);
}

__attribute__((target("avx512f,avx512bw"))) inline __m512i SqDiffAvx512(__m256i a, __m256i b) {
  __m512i x = _mm512_sub_epi16(_mm512_cvtepu8_epi16(a), _mm512_cvtepu8_epi16(b));
  return _mm512_madd_epi16(x, x);
}

__attribute__((target("avx512f,avx512bw"))) inline int64_t HsumAvx512(__m512i v) {
  __m512i lo = _mm512_cvtepi32_epi64(_mm512_castsi512_si256(v));
  __m512i hi = _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(v, 1));
  return _mm512_reduce_add_epi64(_mm512_add_epi64(lo, hi));
}

__attribute__((target("avx512f,avx512bw"))) int64_t L2U8Avx512(const uint8_t* a, const uint8_t* b,
                                                               size_t d) {
  int64_t r = 0;
  size_t i = 0;
  while (i + 32 <= d) {
    __m512i acc = _mm512_setzero_si512();
    size_t end = std::min(d - d % 32, i + kFlushDims);
    for (; i < end; i += 32) {
      acc = _mm512_add_epi32(acc, SqDiffAvx512(_mm256_loadu_si256((const __m256i*)(a + i)),
                                               _mm256_loadu_si256((const __m256i*)(b + i))));
    }
    r += HsumAvx512(acc);
  }
  return r + L2U8Scalar(a + i, b + i, d - i);
}

__attribute__((target("avx512f,avx512bw"))) void L2U8x4Avx512(const uint8_t* q,
                                                              const uint8_t* const* y, size_t d,
                                                              int64_t* out) {
  for (int k = 0; k < 4; k++) out[k] = 0;
  size_t i = 0;
  while (i + 32 <= d) {
    __m512i a0 = _mm512_setzero_si512(), a1 = _mm512_setzero_si512();
    __m512i a2 = _mm512_setzero_si512(), a3 = _mm512_setzero_si512();
    size_t end = std::min(d - d % 32, i + kFlushDims);
    for (; i < end; i += 32) {
      __m256i vq = _mm256_loadu_si256((const __m256i*)(q + i));
      a0 = _mm512_add_epi32(a0, SqDiffAvx512(vq, _mm256_loadu_si256((const __m256i*)(y[0] + i))));
      a1 = _mm512_add_epi32(a1, SqDiffAvx512(vq, _mm256_loadu_si256((const __m256i*)(y[1] + i))));
      a2 = _mm512_add_epi32(a2, SqDiffAvx512(vq, _mm256_loadu_si256((const __m256i*)(y[2] + i))));
      a3 = _mm512_add_epi32(a3, SqDiffAvx512(vq, _mm256_loadu_si256((const __m256i*)(y[3] + i))));
    }
    out[0] += HsumAvx512(a0);
    out[1] += HsumAvx512(a1);
    out[2] += HsumAvx512(a2);
    out[3] += HsumAvx512(a3);
  }
  for (int k = 0; k < 4; k++) out[k] += L2U8Scalar(q + i, y[k] + i, d - i);
}
#endif

struct Kernels {
  L2U8Fn one;
  L2U8x4Fn four;
};

// The kernels of |level| ("avx512", "avx2" or "scalar"), if this CPU has the level.
bool KernelsFor(const std::string& level, Kernels* out) {
#if defined(__x86_64__)
  __builtin_cpu_init();
  if (level == "avx512") {
    if (!__builtin_cpu_supports("avx512bw")) return false;
    *out = {L2U8Avx512, L2U8x4Avx512};
    return true;
  }
  if (level == "avx2") {
    if (!__builtin_cpu_supports("avx2")) return false;
    *out = {L2U8Avx2, L2U8x4Avx2};
    return true;
  }
#endif
  if (level == "scalar") {
    *out = {L2U8Scalar, L2U8x4Scalar};
    return true;
  }
  return false;
}

Kernels PickKernels() {
  Kernels k{L2U8Scalar, L2U8x4Scalar};
  for (const char* level : {"avx512", "avx2"}) {
    if (KernelsFor(level, &k)) break;
  }
  return k;
}

const Kernels& GetKernels() {
  static const Kernels kernels = PickKernels();
  return kernels;
}

// Without a SIMD kernel the 8-bit distance is slower than faiss's float one, so the build stays
// float.
bool HasInt8Kernel() { return GetKernels().one != L2U8Scalar; }

// Clips to the 8-bit range. NaN encodes as 0 and the infinities as the ends, so no value is
// undefined to convert; Encode() turns the 8-bit build off for rows that hold them anyway.
inline uint8_t EncodeValue(float v, float lo, float scale) {
  const float c = (v - lo) * scale;
  if (!(c > 0.0f)) return 0;
  if (c >= 255.0f) return 255;
  return static_cast<uint8_t>(std::nearbyint(c));
}

// The 8-bit range of a batch: its values' 0.1% and 99.9% quantiles, the usual percentile
// calibration, not their extremes. A few outlying values would otherwise stretch the range until
// every other value has only a few codes left, and the build distances between them collapse;
// clipped, they only blur the edges of the rows that hold them. Estimated from a fixed-size sample,
// so the cost does not grow with the batch. Returns false if the values hold no finite one.
constexpr size_t kRangeSamples = 1 << 16;
constexpr double kRangeTail = 1e-3;

bool BatchRange(const float* x, size_t count, float* lo, float* hi) {
  std::vector<float> s;
  if (count <= kRangeSamples) {
    s.assign(x, x + count);
  } else {
    std::mt19937_64 rng(count);  // fixed per batch size: the same rows encode the same way
    std::uniform_int_distribution<size_t> pick(0, count - 1);
    s.resize(kRangeSamples);
    for (auto& v : s) v = x[pick(rng)];
  }
  s.erase(std::remove_if(s.begin(), s.end(), [](float v) { return !std::isfinite(v); }), s.end());
  if (s.empty()) return false;
  const size_t k = static_cast<size_t>(kRangeTail * s.size());
  std::nth_element(s.begin(), s.begin() + k, s.end());
  *lo = s[k];
  std::nth_element(s.begin(), s.end() - 1 - k, s.end());
  *hi = s[s.size() - 1 - k];
  return true;
}

// Normalized rows: every value is in [-1, 1], so their exact extremes bound the range at no more
// than [-1, 1] and leave no tail to clip. A fixed [-1, 1] range is too coarse: a row's values sit
// near 1/sqrt(d), on a few dozen of the 256 levels.
bool ExactRange(const float* x, size_t count, float* lo, float* hi) {
  float mn = std::numeric_limits<float>::infinity(), mx = -std::numeric_limits<float>::infinity();
#pragma omp parallel for reduction(min : mn) reduction(max : mx)
  for (size_t i = 0; i < count; i++) {
    const float v = x[i];
    if (std::isfinite(v)) {
      mn = std::min(mn, v);
      mx = std::max(mx, v);
    }
  }
  if (!(mn <= mx)) return false;
  *lo = mn;
  *hi = mx;
  return true;
}

// Squared L2 over 8-bit codes, scaled back to the units of the float rows.
class Int8L2DistanceComputer : public faiss::DistanceComputer {
 public:
  Int8L2DistanceComputer(const uint8_t* codes, const float* xb, size_t n, size_t d, float lo,
                         float scale)
      : codes_(codes),
        xb_(xb),
        n_(n),
        d_(d),
        lo_(lo),
        scale_(scale),
        inv_(1.0f / (scale * scale)),
        k_(GetKernels()) {}

  void set_query(const float* x) override {
    // The build queries with rows of the storage itself; use their codes as they are.
    if (x >= xb_ && x < xb_ + n_ * d_ && static_cast<size_t>(x - xb_) % d_ == 0) {
      q_ = codes_ + (x - xb_);
      return;
    }
    qbuf_.resize(d_);
    for (size_t i = 0; i < d_; i++) qbuf_[i] = EncodeValue(x[i], lo_, scale_);
    q_ = qbuf_.data();
  }

  float operator()(faiss::idx_t i) override { return k_.one(q_, codes_ + i * d_, d_) * inv_; }

  void distances_batch_4(const faiss::idx_t idx0, const faiss::idx_t idx1, const faiss::idx_t idx2,
                         const faiss::idx_t idx3, float& dis0, float& dis1, float& dis2,
                         float& dis3) override {
    const uint8_t* y[4] = {codes_ + idx0 * d_, codes_ + idx1 * d_, codes_ + idx2 * d_,
                           codes_ + idx3 * d_};
    int64_t r[4];
    k_.four(q_, y, d_, r);
    dis0 = r[0] * inv_;
    dis1 = r[1] * inv_;
    dis2 = r[2] * inv_;
    dis3 = r[3] * inv_;
  }

  float symmetric_dis(faiss::idx_t i, faiss::idx_t j) override {
    return k_.one(codes_ + i * d_, codes_ + j * d_, d_) * inv_;
  }

 private:
  const uint8_t* codes_;
  const float* xb_;
  size_t n_;
  size_t d_;
  float lo_;
  float scale_;
  float inv_;
  const Kernels& k_;
  const uint8_t* q_ = nullptr;
  std::vector<uint8_t> qbuf_;
};

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

namespace hnsw_build_internal {
bool L2U8WithLevel(const char* level, const uint8_t* q, const uint8_t* const* y, size_t d,
                   int64_t* one, int64_t* four) {
  Kernels k;
  if (!KernelsFor(level, &k)) return false;
  *one = k.one(q, y[0], d);
  k.four(q, y, d, four);
  return true;
}
}  // namespace hnsw_build_internal

HnswBuildStorage::HnswBuildStorage(faiss::idx_t d, faiss::MetricType metric,
                                   MemoryAccountantPtr accountant, bool int8_build, bool normalized)
    : faiss::IndexFlat(d, metric), accountant_(std::move(accountant)), normalized_(normalized) {
  // Only squared L2 has an 8-bit kernel; inner product keeps float distances.
  int8_on_ = int8_build && metric == faiss::METRIC_L2 && HasInt8Kernel();
}

HnswBuildStorage::~HnswBuildStorage() { FreeRows(); }

// The index may be destroyed long after its build (an index cache), on any thread, so a throwing
// accountant must not escape the destructor.
void HnswBuildStorage::FreeRows() noexcept {
  codes = faiss::MaybeOwnedVector<uint8_t>();
  uint8_t* rows = rows_;
  uint8_t* int8 = int8_;
  const size_t rows_bytes = capacity_ * code_size, int8_bytes = int8_capacity_ * d;
  rows_ = nullptr;
  int8_ = nullptr;
  capacity_ = 0;
  int8_capacity_ = 0;
  try {
    FreeBlock(int8, int8_bytes, accountant_.get());
    FreeBlock(rows, rows_bytes, accountant_.get());
  } catch (const std::exception& e) {
    std::fprintf(stderr, "tenann: releasing HNSW build storage failed: %s\n", e.what());
  } catch (...) {
    std::fprintf(stderr, "tenann: releasing HNSW build storage failed\n");
  }
}

void HnswBuildStorage::Reserve(size_t rows) {
  if (rows > capacity_) {
    // A mapping grows without copying, so it grows in small steps and is charged little beyond what
    // it holds; the heap may copy on growth, so it doubles.
    const size_t step =
        std::max(accountant_ ? capacity_ / 8 : capacity_, static_cast<size_t>(4096));
    const size_t cap = std::max(rows, capacity_ + step);
    rows_ = GrowBlock(rows_, capacity_ * code_size, cap * code_size, accountant_.get());
    capacity_ = cap;
    SyncView();
  }
  if (int8_on_ && int8_capacity_ < capacity_) {
    try {
      int8_ = GrowBlock(int8_, int8_capacity_ * d, capacity_ * d, accountant_.get());
      int8_capacity_ = capacity_;
    } catch (const std::exception& e) {
      // The 8-bit copy only speeds the build up: without room for it the build goes on in float.
      T_LOG(WARNING) << "HNSW build: no room for the 8-bit copy of the rows (" << e.what()
                     << "), building with float distances";
      DropInt8();
    }
  }
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
  if (int8_on_) Encode(ntotal, ntotal + n);
  ntotal += n;
  SyncView();
}

void HnswBuildStorage::reset() {
  FreeRows();
  ntotal = 0;
  num_staged_ = 0;
  has_range_ = false;
}

faiss::DistanceComputer* HnswBuildStorage::get_distance_computer() const {
  if (int8_on_ && ntotal > 0) {
    int8_used_ = true;
    return new Int8L2DistanceComputer(int8_, xb(), ntotal, d, lo_, scale_);
  }
  return faiss::IndexFlat::get_distance_computer();
}

void HnswBuildStorage::FinishBuild() {
  T_CHECK_EQ(num_staged_, 0) << "staged rows must be added before the build finishes";
  DropInt8();
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

void HnswBuildStorage::DropInt8() noexcept {
  uint8_t* p = int8_;
  const size_t bytes = int8_capacity_ * d;
  int8_ = nullptr;
  int8_capacity_ = 0;
  int8_on_ = false;
  try {
    FreeBlock(p, bytes, accountant_.get());
  } catch (const std::exception& e) {
    std::fprintf(stderr, "tenann: releasing the HNSW 8-bit build copy failed: %s\n", e.what());
  } catch (...) {
    std::fprintf(stderr, "tenann: releasing the HNSW 8-bit build copy failed\n");
  }
}

bool HnswBuildStorage::Int8BuildSupported() { return HasInt8Kernel(); }

void HnswBuildStorage::Encode(size_t from, size_t to) {
  float mn, mx;
  const float* x = xb() + from * d;
  const size_t count = (to - from) * d;
  if (!(normalized_ ? ExactRange(x, count, &mn, &mx) : BatchRange(x, count, &mn, &mx))) {
    T_LOG(WARNING) << "HNSW build: rows hold no finite value, building with float distances";
    DropInt8();
    return;
  }
  const float hi = lo_ + 255.0f / scale_;
  if (!has_range_ || mn < lo_ || mx > hi) {
    // Widen to cover both ranges and re-encode the rows already coded. A widening grows the span
    // by at least half, so a range that keeps drifting re-encodes the rows a few times per
    // tenfold growth, not once per batch.
    float new_lo = has_range_ ? std::min(mn, lo_) : mn;
    float new_hi = has_range_ ? std::max(mx, hi) : mx;
    const float span = std::max(new_hi - new_lo, 1e-20f);
    float margin = 0.05f * span;
    if (has_range_) margin = std::max(margin, 0.5f * (1.5f * (hi - lo_) - span));
    new_lo -= margin;
    new_hi += margin;
    if (!(std::isfinite(new_lo) && std::isfinite(new_hi) && new_hi > new_lo)) {
      T_LOG(WARNING) << "HNSW build: values too large for an 8-bit range, building with float "
                        "distances";
      DropInt8();
      return;
    }
    lo_ = new_lo;
    scale_ = 255.0f / (new_hi - new_lo);
    has_range_ = true;
    from = 0;
  }
  if (!EncodeRows(from, to)) {
    T_LOG(WARNING) << "HNSW build: a row holds a non-finite value, building with float distances";
    DropInt8();
  }
}

bool HnswBuildStorage::EncodeRows(size_t from, size_t to) {
  const float* x = xb();
  const float lo = lo_, scale = scale_;
  int bad = 0;
#pragma omp parallel for reduction(| : bad)
  for (size_t r = from; r < to; r++) {
    for (size_t i = 0; i < static_cast<size_t>(d); i++) {
      const float v = x[r * d + i];
      bad |= !std::isfinite(v);
      int8_[r * d + i] = EncodeValue(v, lo, scale);
    }
  }
  return bad == 0;
}

}  // namespace tenann
