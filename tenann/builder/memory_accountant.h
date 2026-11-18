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

#pragma once

#include <cstdint>
#include <memory>

// Present when IndexBuilder has SetMemoryAccountant().
#define TENANN_HAS_BUILD_MEMORY_ACCOUNTANT 1

namespace tenann {

/// Accounts memory a builder maps itself. Mapped memory bypasses the caller's allocator hooks, so
/// the builder reports it here instead: Charge() before bytes are mapped, Release() after they are
/// unmapped. Accounting only; a builder never declines an allocation on its account. The index
/// can outlive its builder (an index cache), and releases what it still holds when destroyed.
class MemoryAccountant {
 public:
  virtual ~MemoryAccountant() = default;
  virtual void Charge(int64_t bytes) = 0;
  virtual void Release(int64_t bytes) = 0;
};

using MemoryAccountantPtr = std::shared_ptr<MemoryAccountant>;

}  // namespace tenann
