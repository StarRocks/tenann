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

#include <cstdint>
#include <random>
#include <vector>

#include "gtest/gtest.h"
#include "tenann/index/internal/index_ivfpq.h"

namespace tenann {

namespace {

/// Table layout the kernels assume: row m holds the pq_ksub sub-distances for
/// subquantizer m, so a code byte indexes within its own row.
std::vector<float> MakeSimTable(size_t pq_m, size_t pq_ksub, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-4.0F, 4.0F);
  std::vector<float> table(pq_m * pq_ksub);
  for (auto& v : table) {
    v = dist(rng);
  }
  return table;
}

std::vector<uint8_t> MakeCodes(size_t pq_m, size_t pq_ksub, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> dist(0, static_cast<int>(pq_ksub) - 1);
  std::vector<uint8_t> codes(pq_m);
  for (auto& c : codes) {
    c = static_cast<uint8_t>(dist(rng));
  }
  return codes;
}

/// Independent restatement of the kernel contract, written the obvious way.
float ReferenceSum(const std::vector<uint8_t>& codes, const std::vector<float>& table,
                   size_t pq_ksub) {
  double result = 0;
  for (size_t m = 0; m < codes.size(); m++) {
    result += table[m * pq_ksub + codes[m]];
  }
  return static_cast<float>(result);
}

// pq_m values chosen to hit every shape of the AVX2 loop: below one 16-wide block,
// exactly one block, one block plus leftovers, two blocks, two blocks plus leftovers.
const std::vector<size_t> kPqMValues = {1, 8, 15, 16, 17, 20, 32, 33, 48, 64};

}  // namespace

// The generic kernel must agree with a straightforward reference implementation.
// Without this, an equivalence-only test would pass even if both kernels shared the
// same wrong idea of the table layout.
TEST(IvfPqPq8SimdTest, GenericMatchesReference) {
  constexpr size_t kPqKsub = 256;
  for (size_t pq_m : kPqMValues) {
    auto table = MakeSimTable(pq_m, kPqKsub, /*seed=*/1234 + pq_m);
    auto codes = MakeCodes(pq_m, kPqKsub, /*seed=*/5678 + pq_m);

    float actual =
        ivfpq_simd::Pq8DistanceSingleCodeGeneric(codes.data(), table.data(), pq_m, kPqKsub);
    EXPECT_NEAR(actual, ReferenceSum(codes, table, kPqKsub), 1e-3)
        << "generic kernel disagrees at pq_m=" << pq_m;
  }
}

TEST(IvfPqPq8SimdTest, GenericHandlesHandCheckedCase) {
  // Two subquantizers, four centroids each. Row 0 is {0, 1, 2, 3}, row 1 is {4, 5, 6, 7}.
  const std::vector<float> table = {0, 1, 2, 3, 4, 5, 6, 7};
  const std::vector<uint8_t> codes = {2, 1};
  // Expected: table[0 * 4 + 2] + table[1 * 4 + 1] = 2 + 5 = 7.
  EXPECT_FLOAT_EQ(
      ivfpq_simd::Pq8DistanceSingleCodeGeneric(codes.data(), table.data(), codes.size(), 4), 7.0F);
}

#if defined(__x86_64__)
// The AVX2 kernel gathers 16 codes per iteration and sums lane-wise, so it does not
// accumulate in the same order as the generic kernel. It must still land on the same
// value within float rounding, at every loop shape.
TEST(IvfPqPq8SimdTest, Avx2MatchesGeneric) {
  if (!ivfpq_simd::Avx2Supported()) {
    GTEST_SKIP() << "CPU has no AVX2; the AVX2 kernel is unreachable here";
  }

  constexpr size_t kPqKsub = 256;
  for (size_t pq_m : kPqMValues) {
    auto table = MakeSimTable(pq_m, kPqKsub, /*seed=*/4321 + pq_m);
    auto codes = MakeCodes(pq_m, kPqKsub, /*seed=*/8765 + pq_m);

    float generic =
        ivfpq_simd::Pq8DistanceSingleCodeGeneric(codes.data(), table.data(), pq_m, kPqKsub);
    float avx2 = ivfpq_simd::Pq8DistanceSingleCodeAvx2(codes.data(), table.data(), pq_m, kPqKsub);

    EXPECT_NEAR(avx2, generic, 1e-3) << "AVX2 kernel disagrees at pq_m=" << pq_m;
  }
}

// pq_ksub is the row stride, so a non-256 value catches a kernel that hard-codes it.
TEST(IvfPqPq8SimdTest, Avx2HonoursRowStride) {
  if (!ivfpq_simd::Avx2Supported()) {
    GTEST_SKIP() << "CPU has no AVX2; the AVX2 kernel is unreachable here";
  }

  for (size_t pq_ksub : {16U, 64U, 256U}) {
    const size_t pq_m = 33;
    auto table = MakeSimTable(pq_m, pq_ksub, /*seed=*/99);
    auto codes = MakeCodes(pq_m, pq_ksub, /*seed=*/101);

    float generic =
        ivfpq_simd::Pq8DistanceSingleCodeGeneric(codes.data(), table.data(), pq_m, pq_ksub);
    float avx2 = ivfpq_simd::Pq8DistanceSingleCodeAvx2(codes.data(), table.data(), pq_m, pq_ksub);

    EXPECT_NEAR(avx2, generic, 1e-3) << "kernels disagree at pq_ksub=" << pq_ksub;
    EXPECT_NEAR(generic, ReferenceSum(codes, table, pq_ksub), 1e-3);
  }
}
#endif  // __x86_64__

// Whatever the CPU offers, the dispatcher must return the same answer as the generic
// kernel. This is the entry point the IVF-PQ scanner actually calls.
TEST(IvfPqPq8SimdTest, DispatcherMatchesGeneric) {
  constexpr size_t kPqKsub = 256;
  for (size_t pq_m : kPqMValues) {
    auto table = MakeSimTable(pq_m, kPqKsub, /*seed=*/2468 + pq_m);
    auto codes = MakeCodes(pq_m, kPqKsub, /*seed=*/1357 + pq_m);

    float dispatched = ivfpq_simd::Pq8DistanceSingleCode(codes.data(), table.data(), pq_m, kPqKsub);
    float generic =
        ivfpq_simd::Pq8DistanceSingleCodeGeneric(codes.data(), table.data(), pq_m, kPqKsub);

    EXPECT_NEAR(dispatched, generic, 1e-3) << "dispatcher disagrees at pq_m=" << pq_m;
  }
}

}  // namespace tenann
