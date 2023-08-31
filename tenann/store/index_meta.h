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

#include "tenann/common/json.hpp"
#include "tenann/store/index_type.h"

namespace tenann {

using json = nlohmann::json;

class IndexMeta {
 public:
  IndexMeta();
  explicit IndexMeta(const json& meta_json);

  IndexMeta(const IndexMeta&) = default;
  IndexMeta& operator=(const IndexMeta&) = default;
  IndexMeta(IndexMeta&&) noexcept = default;
  IndexMeta& operator=(IndexMeta&&) noexcept = default;

  json& meta_json();
  const json& meta_json() const;

  /* setters and getters */
  void SetMetaVersion(int version);
  void SetIndexFamily(IndexFamily family);
  void SetIndexType(IndexType type);

  int meta_version() const;
  int index_family() const;
  int index_type() const;

  json& common_params();
  json& index_params();
  json& search_params();
  json& extra_params();

  const json& common_params() const;
  const json& index_params() const;
  const json& search_params() const;
  const json& extra_params() const;

  /// Read from a json file
  static IndexMeta Read(const std::string& path);

  /// Deserialize from a binary buffer (using the MessagePack format)
  static IndexMeta Deserialize(const std::vector<uint8_t>& buffer);

  /// Write to a json wile
  void Write(const std::string& path);

  /// Serialize to a binary buffer (using the MessagePack format)
  std::vector<uint8_t> Serialize();

  /// Check meta data integrity and save possible error message into [[err_msg]]
  bool CheckIntegrity(std::string* err_msg);

 private:
  json meta_json_;
};

}  // namespace tenann