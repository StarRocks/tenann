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

#include <cstdlib>
#include <set>
#include <string>

#include "gtest/gtest.h"
#include "tenann/common/simd_info.h"

namespace tenann {

TEST(SimdInfoTest, ReportsAKnownLevel) {
  const std::set<std::string> kKnown = {"NONE",     "AVX2",    "AVX512",   "AVX512_SPR",
                                        "ARM_NEON", "ARM_SVE", "RISCV_RVV"};
  const std::string level = SimdLevelName();
  EXPECT_FALSE(level.empty());
  EXPECT_EQ(kKnown.count(level), 1U) << "unrecognized SIMD level name: " << level;
}

#if defined(__x86_64__)
namespace {

bool CpuHasAvx512() {
  return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512dq") &&
         __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512vl");
}

bool CpuHasAvx2() { return __builtin_cpu_supports("avx2"); }

/// FAISS_SIMD_LEVEL pins the level, which is how a benchmark compares two of them.
bool LevelIsPinned() {
  const char* env = std::getenv("FAISS_SIMD_LEVEL");
  return env != nullptr && env[0] != '\0';
}

}  // namespace

// Reporting a level the CPU cannot execute is the one outcome that faults rather than
// merely running slowly, so it must hold no matter how the level was chosen.
TEST(SimdInfoTest, LevelNeverExceedsCpuCapability) {
  const std::string level = SimdLevelName();

  if (!CpuHasAvx512()) {
    EXPECT_NE(level.rfind("AVX512", 0), 0U)
        << "kernels report " << level << " on a CPU without AVX-512; this would fault";
  }
  if (!CpuHasAvx2()) {
    EXPECT_EQ(level, "NONE") << "kernels report " << level
                             << " on a CPU without AVX2; this would fault";
  }
}

// Left to choose for itself, the dispatch must land on the widest level the CPU
// offers. Anything less means it is not really dispatching -- most likely because
// faiss was built at a fixed FAISS_OPT_LEVEL instead of "dd", which silently leaves
// the wider kernels out of the library altogether.
TEST(SimdInfoTest, PicksTheWidestLevelTheCpuOffers) {
  if (LevelIsPinned()) {
    GTEST_SKIP() << "FAISS_SIMD_LEVEL pins the level, so there is no choice to check";
  }

  const std::string level = SimdLevelName();
  if (CpuHasAvx512()) {
    EXPECT_EQ(level.rfind("AVX512", 0), 0U)
        << "CPU supports AVX-512 but the kernels report " << level
        << "; faiss is probably not built with FAISS_OPT_LEVEL=dd";
  } else if (CpuHasAvx2()) {
    EXPECT_EQ(level, "AVX2") << "CPU supports AVX2 but the kernels report " << level;
  } else {
    EXPECT_EQ(level, "NONE");
  }
}
#endif  // __x86_64__

}  // namespace tenann
