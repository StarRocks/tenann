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

#include "tenann/common/json.hpp"
#include "tenann/index/index.h"

namespace tenann {

class IndexReader {
 public:
  explicit IndexReader(const IndexMeta& meta) : index_meta_(meta){};
  virtual ~IndexReader();

  T_FORBID_DEFAULT_CTOR(IndexReader);
  T_FORBID_COPY_AND_ASSIGN(IndexReader);
  T_FORBID_MOVE(IndexReader);

  // Read index file
  virtual IndexRef ReadIndex(const std::string& path) = 0;

  nlohmann::json& conf();
  const nlohmann::json& conf() const;

  /** Getters */
  const IndexMeta& index_meta() const;

 protected:
  /// @brief index meta
  IndexMeta index_meta_;

  /// @brief read options
  nlohmann::json conf_;
};

}  // namespace tenann
