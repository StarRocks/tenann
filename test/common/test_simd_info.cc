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

#if defined(__aarch64__)
#include <sys/auxv.h>
#ifndef HWCAP_SVE
#define HWCAP_SVE (1 << 22)
#endif
#endif

extern "C" char* openblas_get_config(void);

namespace {

/// FAISS_SIMD_LEVEL pins the level, which is how a benchmark compares two of them.
bool LevelIsPinned() {
  const char* env = std::getenv("FAISS_SIMD_LEVEL");
  return env != nullptr && env[0] != '\0';
}

}  // namespace

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

// This library is built at a fixed FAISS_OPT_LEVEL, so the level is decided at compile
// time and does not follow the CPU. Asserting that it takes the widest level available
// would be asserting the dynamic-dispatch contract, which this build deliberately does
// not have -- see build-thirdparty.sh for why "dd" is not used.
TEST(SimdInfoTest, LevelIsFixedAtBuildTimeNotChosenPerCpu) {
  if (LevelIsPinned()) {
    GTEST_SKIP() << "FAISS_SIMD_LEVEL pins the level explicitly";
  }
  const std::string level = SimdLevelName();
  // The x86 build targets AVX2. A wider level here would mean kernels the baseline
  // package must not contain.
  EXPECT_TRUE(level == "AVX2" || level == "NONE")
      << "fixed-level build reports " << level << "; expected the compiled-in level";
}
#endif  // __x86_64__

#if defined(__aarch64__)
namespace {

bool CpuHasSve() { return (getauxval(AT_HWCAP) & HWCAP_SVE) != 0; }

}  // namespace

// The ARM counterpart of the x86 invariant, and the one that decides whether a single
// package can serve every machine: reporting ARM_SVE where the CPU has none faults.
TEST(SimdInfoTest, LevelNeverExceedsCpuCapability) {
  if (!CpuHasSve()) {
    EXPECT_NE(SimdLevelName(), "ARM_SVE")
        << "kernels report ARM_SVE on a CPU without SVE; this would fault";
  }
}

// The aarch64 build is fixed at the generic level, so NEON is what the kernels report
// whether or not the CPU has SVE. The SVE package is a separate build.
TEST(SimdInfoTest, LevelIsFixedAtBuildTimeNotChosenPerCpu) {
  if (LevelIsPinned()) {
    GTEST_SKIP() << "FAISS_SIMD_LEVEL pins the level explicitly";
  }
  const std::string level = SimdLevelName();
  EXPECT_TRUE(level == "ARM_NEON" || level == "NONE")
      << "fixed-level build reports " << level << "; expected the compiled-in level";
}
#endif  // __aarch64__

// A DYNAMIC_ARCH OpenBLAS picks its kernels the way faiss picks its own. Linking a
// single-architecture build instead is silent: it runs, and it either leaves the wide
// kernels unused or faults on an older CPU.
TEST(SimdInfoTest, OpenBlasDispatchesAtRunTime) {
  const std::string config = openblas_get_config();
  EXPECT_NE(config.find("DYNAMIC_ARCH"), std::string::npos)
      << "OpenBLAS reports \"" << config << "\"; a single-architecture build was linked";
}

}  // namespace tenann
