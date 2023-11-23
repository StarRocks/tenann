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

#include "tenann/store/index_meta.h"

#include <fstream>

#include "index_meta.h"
#include "tenann/common/logging.h"

namespace tenann {

IndexMeta::IndexMeta() {
  meta_json_[kCommonKey] = {};
  meta_json_[kIndexKey] = {};
  meta_json_[kSearchKey] = {};
  meta_json_[kExtraKey] = {};
};

IndexMeta::IndexMeta(const json& meta_json) : meta_json_(meta_json) {}

json& IndexMeta::meta_json() { return meta_json_; }

const json& IndexMeta::meta_json() const { return meta_json_; }

void IndexMeta::SetMetaVersion(int version) { meta_json_["meta_version"] = version; }
void IndexMeta::SetIndexFamily(tenann::IndexFamily family) { meta_json_["family"] = family; }
void IndexMeta::SetIndexType(tenann::IndexType type) { meta_json_["type"] = type; }

int IndexMeta::meta_version() const {
  if (!meta_json_.contains("meta_version")) {
    T_LOG(ERROR) << "meta_version (`meta_versoin`) not set in index meta";
  }
  return meta_json_["meta_version"].template get<int>();
}

int IndexMeta::index_family() const {
  if (!meta_json_.contains("family")) {
    T_LOG(ERROR) << "index faimily not set in index meta";
  }
  return meta_json_["family"].template get<int>();
}

int IndexMeta::index_type() const {
  if (!meta_json_.contains("type")) {
    T_LOG(ERROR) << "index type not set in index meta";
  }
  return meta_json_["type"].template get<int>();
}

json& IndexMeta::common_params() { return meta_json_["common"]; }
json& IndexMeta::index_params() { return meta_json_["index"]; }
json& IndexMeta::search_params() { return meta_json_["search"]; }
json& IndexMeta::extra_params() { return meta_json_["extra"]; }

const json& IndexMeta::common_params() const { return meta_json_["common"]; }
const json& IndexMeta::index_params() const { return meta_json_["index"]; }
const json& IndexMeta::search_params() const { return meta_json_["search"]; }
const json& IndexMeta::extra_params() const { return meta_json_["extra"]; }

IndexMeta IndexMeta::Read(const std::string& file_path) {
  try {
    std::ifstream input_file(file_path);
    if (!input_file.is_open()) {
      T_LOG(ERROR) << "Failed to open file: " << file_path << std::endl;
      return IndexMeta();
    }

    nlohmann::json json_obj;
    input_file >> json_obj;
    input_file.close();
    IndexMeta meta(json_obj);
    meta.CheckOrThrowError();
    return meta;
  } catch (json::exception& e) {
    T_LOG(ERROR) << e.what();
  }
}

IndexMeta IndexMeta::Deserialize(const std::vector<uint8_t>& buffer) {
  try {
    auto meta_json = json::from_msgpack(buffer);
    return IndexMeta(meta_json);
  } catch (json::exception& e) {
    T_LOG(ERROR) << e.what();
  }
}

IndexMeta IndexMeta::Parse(const std::string& str) {
  try {
    auto meta_json = json::parse(str);
    IndexMeta meta(meta_json);
    meta.CheckOrThrowError();
    return meta;
  } catch (json::exception& e) {
    T_LOG(ERROR) << e.what();
  }
}

bool IndexMeta::Write(const std::string& file_path) {
  try {
    std::ofstream output_file(file_path);
    if (!output_file.is_open()) {
      T_LOG(ERROR) << "Failed to open file: " << file_path << std::endl;
      return false;
    }

    output_file << meta_json_;
    output_file.close();
  } catch (json::exception& e) {
    T_LOG(ERROR) << e.what();
  }
  return true;
}

std::vector<uint8_t> IndexMeta::Serialize() {
  try {
    return json::to_msgpack(meta_json_);
  } catch (json::exception& e) {
    T_LOG(ERROR) << e.what();
  }
}

std::string IndexMeta::Stringify(int indent) {
  try {
    return meta_json_.dump(indent);
  } catch (json::exception& e) {
    T_LOG(ERROR) << e.what();
  }
}

bool IndexMeta::CheckIntegrity(std::string* err_msg) noexcept {
  if (!meta_json_.contains("meta_version")) {
    *err_msg = "meta_version (`meta_versoin`) not set in index meta";
    return false;
  }

  if (!meta_json_.contains("family")) {
    *err_msg = "index faimily not set in index meta";
    return false;
  }

  if (!meta_json_.contains("type")) {
    *err_msg = "index type not set in index meta";
    return false;
  }

  return true;
}

void IndexMeta::CheckOrThrowError() T_THROW_EXCEPTION {
  std::string msg;
  if (!CheckIntegrity(&msg)) {
    T_LOG(ERROR) << msg;
  }
}

}  // namespace tenann