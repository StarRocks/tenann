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
#include <optional>
#include <sstream>
#include <type_traits>

#include "fmt/format.h"
#include "tenann/common/error.h"
#include "tenann/common/json.h"
#include "tenann/common/macros.h"
#include "tenann/store/index_type.h"

namespace tenann {

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
  json& index_writer_options();
  json& index_reader_options();

  const json& common_params() const;
  const json& index_params() const;
  const json& search_params() const;
  const json& extra_params() const;
  const json& index_writer_options() const;
  const json& index_reader_options() const;

  /// Read from a json file
  static IndexMeta Read(const std::string& path);

  /// Deserialize from a binary buffer (using the MessagePack format)
  static IndexMeta Deserialize(const std::vector<uint8_t>& buffer);

  /// Parse index meta from a json string
  static IndexMeta Parse(const std::string& str);

  /// Write to a json wile
  bool Write(const std::string& path);

  /// Serialize to a binary buffer (using the MessagePack format)
  std::vector<uint8_t> Serialize();

  /// Converts an IndexMeta object to a JSON string
  std::string Stringify(int indent = -1);

  /// Check meta data integrity and save possible error message into [[err_msg]]
  bool CheckIntegrity(std::string* err_msg) noexcept;

  void CheckOrThrowError() T_THROW_EXCEPTION;

  template <typename T, typename = std::enable_if_t<!std::is_reference_v<T>>>
  T GetRequiredCommonParam(const char* key) const {
    return GetRequired<T>(kCommonKey, key);
  }

  template <typename T, typename = std::enable_if_t<!std::is_reference_v<T>>>
  T GetRequiredIndexParam(const char* key) const {
    return GetRequired<T>(kIndexKey, key);
  }

  template <typename T, typename = std::enable_if_t<!std::is_reference_v<T>>>
  T GetRequiredSearchParam(const char* key) const {
    return GetRequired<T>(kSearchKey, key);
  }

  template <typename T, typename = std::enable_if_t<!std::is_reference_v<T>>>
  T GetRequiredExtraParam(const char* key) const {
    return GetRequired<T>(kExtraKey, key);
  }

  template <typename T, typename = std::enable_if_t<!std::is_reference_v<T>>>
  T GetRequiredWriteIndexParam(const char* key) const {
    return GetRequired<T>(kWriteIndexKey, key);
  }

  template <typename T, typename = std::enable_if_t<!std::is_reference_v<T>>>
  T GetRequiredReadIndexParam(const char* key) const {
    return GetRequired<T>(kReadIndexKey, key);
  }

  template <typename T, typename = std::enable_if_t<!std::is_reference_v<T>>>
  std::optional<T> GetOptionalCommonParam(const char* key) const {
    return GetOptional<T>(kCommonKey, key);
  }

  template <typename T, typename = std::enable_if_t<!std::is_reference_v<T>>>
  std::optional<T> GetOptionalIndexParam(const char* key) const {
    return GetOptional<T>(kIndexKey, key);
  }

  template <typename T, typename = std::enable_if_t<!std::is_reference_v<T>>>
  std::optional<T> GetOptionalSearchParam(const char* key) const {
    return GetOptional<T>(kSearchKey, key);
  }

  template <typename T, typename = std::enable_if_t<!std::is_reference_v<T>>>
  std::optional<T> GetOptionalExtraParam(const char* key) const {
    return GetOptional<T>(kExtraKey, key);
  }

  template <typename T, typename = std::enable_if_t<!std::is_reference_v<T>>>
  std::optional<T> GetOptionalWriteIndexParam(const char* key) const {
    return GetOptional<T>(kWriteIndexKey, key);
  }

  template <typename T, typename = std::enable_if_t<!std::is_reference_v<T>>>
  std::optional<T> GetOptionalReadIndexParam(const char* key) const {
    return GetOptional<T>(kReadIndexKey, key);
  }

 private:
  static constexpr const char* kCommonKey = "common";
  static constexpr const char* kIndexKey = "index";
  static constexpr const char* kSearchKey = "search";
  static constexpr const char* kExtraKey = "extra";
  static constexpr const char* kWriteIndexKey = "write_index";
  static constexpr const char* kReadIndexKey = "read_index";

  template <typename T, typename = std::enable_if_t<!std::is_reference_v<T>>>
  std::optional<T> GetOptional(const char* section_key, const char* key) const {
    try {
      auto sub_json = meta_json_[section_key];
      if (sub_json.contains(key)) {
        auto value = sub_json[key].template get<T>();
        return value;
      }
      return std::nullopt;
    } catch (json::exception& e) {
      throw Error(__FILE__, __LINE__, e.what());
    }
  };

  template <typename T, typename = std::enable_if_t<!std::is_reference_v<T>>>
  T GetRequired(const char* section_key, const char* key) const {
    try {
      auto sub_json = meta_json_[section_key];
      if (!sub_json.contains(key)) {
        std::ostringstream oss;
        oss << "required " << section_key << " parameter `" << key << "` is not set in IndexMeta";
        throw Error(__FILE__, __LINE__, oss.str());
      }
      auto value = sub_json[key].template get<T>();
      return value;
    } catch (json::exception& e) {
      throw Error(__FILE__, __LINE__, e.what());
    }
  };

  json meta_json_;
};

// TODO: delete this macro and its use cases
#define CHECK_AND_GET_META(meta, section, name, type, result)                                    \
  if (!(meta).section##_params().contains(name)) {                                               \
    T_LOG(ERROR) << fmt::format("required {} parameter `{}` is not set in index meta", #section, \
                                name);                                                           \
  } else {                                                                                       \
    result = (meta).section##_params()[name].get<type>();                                        \
  }

// TODO: delete this macro and its use cases
#define GET_META_OR_DEFAULT(meta, section, name, type, result, default_value) \
  if (!(meta).section##_params().contains(name)) {                            \
    result = default_value;                                                   \
  } else {                                                                    \
    result = (meta).section##_params()[name].get<type>();                     \
  }
}  // namespace tenann