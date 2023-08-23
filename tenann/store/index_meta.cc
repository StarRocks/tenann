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

json& IndexMeta::meta_json() { return meta_json_; }

const json& IndexMeta::meta_json() const { return meta_json_; }

void IndexMeta::SetMetaVersion(int version) { meta_json_["meta_version"] = version; }
void IndexMeta::SetIndexFamily(tenann::IndexFamily family) { meta_json_["family"] = family; }
void IndexMeta::SetIndexType(tenann::IndexType type) { meta_json_["type"] = type; }

int IndexMeta::meta_version() const { return meta_json_["meta_version"].template get<int>(); }
int IndexMeta::index_family() const { return meta_json_["family"].template get<int>(); }
int IndexMeta::index_type() const { return meta_json_["type"].template get<int>(); }

json& IndexMeta::common_params() { return meta_json_["common"]; }
json& IndexMeta::index_params() { return meta_json_["index"]; }
json& IndexMeta::search_params() { return meta_json_["search"]; }
json& IndexMeta::extra_params() { return meta_json_["extra"]; }

const json& IndexMeta::common_params() const { return meta_json_["common"]; }
const json& IndexMeta::index_params() const { return meta_json_["index"]; }
const json& IndexMeta::search_params() const { return meta_json_["search"]; }
const json& IndexMeta::extra_params() const { return meta_json_["extra"]; }

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