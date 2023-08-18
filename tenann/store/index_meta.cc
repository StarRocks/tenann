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

namespace tenann {

IndexMeta::IndexMeta() = default;

IndexMeta::IndexMeta(const json& meta_json) : meta_json_(meta_json) {}

void IndexMeta::SetMetaFormatVersion(int version) { meta_json_["meta_format_version"] = version; }

void IndexMeta::SetIndexFamily(tenann::IndexFamily family) { meta_json_["index_family"] = family; }

int IndexMeta::meta_format_version() {
  return meta_json_["meta_format_version"].template get<int>();
}

int IndexMeta::index_family() { return meta_json_["index_family"].template get<int>(); }

json& IndexMeta::meta_json() { return meta_json_; }
json& IndexMeta::common_params() { return meta_json_["common_params"]; }
json& IndexMeta::index_params() { return meta_json_["index_params"]; }
json& IndexMeta::search_params() { return meta_json_["search_params"]; }
json& IndexMeta::extra_properties() { return meta_json_["extra_properties"]; }

// @TODO
IndexMeta IndexMeta::Read(const std::string& path) {}

IndexMeta IndexMeta::Deserialize(const std::vector<uint8_t>& buffer) {
  auto meta_json = json::from_msgpack(buffer);
  return IndexMeta(meta_json);
}

// @TODO
void IndexMeta::Write(const std::string& path) {}

std::vector<uint8_t> IndexMeta::Serialize() { return json::to_msgpack(meta_json_); }

// @TODO
bool IndexMeta::CheckIntegrity(std::string* err_msg) { return true; }

}  // namespace tenann