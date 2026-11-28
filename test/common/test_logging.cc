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

#include <atomic>
#include <thread>
#include <vector>

#include "gtest/gtest.h"
#include "tenann/common/logging.h"

namespace tenann {

TEST(LoggingTest, ConcurrentLogInfoAndWarning) {
  constexpr int kNumThreads = 8;
  constexpr int kIterations = 50;

  std::atomic<int> ready{0};
  std::atomic<bool> start{false};
  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);

  for (int t = 0; t < kNumThreads; ++t) {
    threads.emplace_back([t, &ready, &start]() {
      ready.fetch_add(1, std::memory_order_acq_rel);
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      for (int i = 0; i < kIterations; ++i) {
        if ((i & 1) == 0) {
          T_LOG(INFO) << "thread=" << t << " iter=" << i;
        } else {
          T_LOG(WARNING) << "thread=" << t << " iter=" << i;
        }
      }
    });
  }

  while (ready.load(std::memory_order_acquire) < kNumThreads) {
    std::this_thread::yield();
  }
  start.store(true, std::memory_order_release);

  for (auto& thread : threads) {
    thread.join();
  }
}

}  // namespace tenann
