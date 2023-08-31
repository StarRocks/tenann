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

#include "tenann/index/index.h"

#include "index.h"

namespace tenann {

Index::~Index() noexcept {
  if (index_raw_ != nullptr) {
    deleter_(index_raw_);
  }
}

Index::Index(Index&& rhs) noexcept { std::swap(*this, rhs); }

Index::Index(void* index, IndexType index_type, const std::function<void(void*)>& deleter) noexcept
    : index_raw_(index), index_type_(index_type), deleter_(deleter) {}

Index& Index::operator=(Index&& rhs) noexcept {
  std::swap(*this, rhs);
  return *this;
}

void Index::SetIndexRaw(void* index) { index_raw_ = index; }

void Index::SetIndexType(IndexType index_type) { index_type_ = index_type; }

void* Index::index_raw() const { return index_raw_; }

IndexType Index::index_type() const { return index_type_; }

// @TODO(petri): implement it
size_t Index::EstimateMemorySizeInBytes() { return 1; }

}  // namespace tenann