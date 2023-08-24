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

#include <functional>
#include <memory>
#include <utility>

#include "tenann/common/macros.h"
#include "tenann/store/index_meta.h"

namespace tenann {

class Index {
 public:
  Index(void* index_raw, IndexType index_type,
        const std::function<void(void* index)>& deleter) noexcept;

  ~Index() noexcept;

  // disable copy, enable moving
  T_FORBID_COPY_AND_ASSIGN(Index);

  Index(Index&& rhs) noexcept;

  Index& operator=(Index&& rhs) noexcept;

  /* setters and getters */
  void SetIndexRaw(void* index);
  void SetIndexType(IndexType index_type);

  void* index_raw() const;
  IndexType index_type() const;


  /**
   * @brief  Get the amount of memory occupied by the index in bytes.
   * 
   * @return size_t 
   * 
   * @note Currently this function always return 1.
   * 
   * @TODO(petri): implement it
   */
  size_t EstimateMemorySizeInBytes();

 private:
  void* index_raw_;
  IndexType index_type_;
  std::function<void(void* index_raw)> deleter_;
};

using IndexRef = std::shared_ptr<Index>;

}  // namespace tenann
