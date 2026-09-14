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

#include "faiss/impl/pq_code_distance/pq_code_distance-inl.h"
#include "gtest/gtest.h"

namespace tenann {

namespace {

/// Table layout faiss assumes: row m holds the 256 sub-distances for subquantizer m,
/// so a code byte indexes within its own row. ksub is 1 << nbits, hence always 256
/// for 8-bit codes, which is why faiss takes no stride argument.
constexpr size_t kPqKsub = 256;

std::vector<float> MakeSimTable(size_t pq_m, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-4.0F, 4.0F);
  std::vector<float> table(pq_m * kPqKsub);
  for (auto& v : table) {
    v = dist(rng);
  }
  return table;
}

std::vector<uint8_t> MakeCodes(size_t pq_m, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> dist(0, static_cast<int>(kPqKsub) - 1);
  std::vector<uint8_t> codes(pq_m);
  for (auto& c : codes) {
    c = static_cast<uint8_t>(dist(rng));
  }
  return codes;
}

/// Independent restatement of the contract, written the obvious way.
float ReferenceSum(const std::vector<uint8_t>& codes, const std::vector<float>& table) {
  double result = 0;
  for (size_t m = 0; m < codes.size(); m++) {
    result += table[m * kPqKsub + codes[m]];
  }
  return static_cast<float>(result);
}

// Subquantizer counts spanning every shape faiss's kernels branch on: below one
// vector block, exactly one, one plus leftovers, and the m=4 / m=8 special cases.
const std::vector<size_t> kPqMValues = {1, 4, 8, 15, 16, 17, 20, 32, 33, 48, 64};

}  // namespace

// The IVF-PQ scanner hands faiss (M, sim_table, code). Swapping an argument or
// misreading the row-major layout would still compile and still return a float, so
// pin the contract against an independent reference. This is what the scanner's
// PQDecoder8 overload depends on; the kernels themselves are faiss's to test.
TEST(IvfPqPq8SimdTest, FaissKernelMatchesReference) {
  for (size_t pq_m : kPqMValues) {
    auto table = MakeSimTable(pq_m, /*seed=*/1234 + pq_m);
    auto codes = MakeCodes(pq_m, /*seed=*/5678 + pq_m);

    float actual = faiss::pq_code_distance_8bit_single(pq_m, table.data(), codes.data());
    EXPECT_NEAR(actual, ReferenceSum(codes, table), 1e-3)
        << "faiss PQ8 kernel disagrees at pq_m=" << pq_m;
  }
}

TEST(IvfPqPq8SimdTest, FaissKernelHandlesHandCheckedCase) {
  // Two subquantizers. Row 0 is {0, 1, 2, ...}, row 1 is {256, 257, 258, ...}.
  std::vector<float> table(2 * kPqKsub);
  for (size_t i = 0; i < table.size(); i++) {
    table[i] = static_cast<float>(i);
  }
  const std::vector<uint8_t> codes = {2, 1};
  // Expected: table[0 * 256 + 2] + table[1 * 256 + 1] = 2 + 257 = 259.
  EXPECT_FLOAT_EQ(faiss::pq_code_distance_8bit_single(codes.size(), table.data(), codes.data()),
                  259.0F);
}

}  // namespace tenann
