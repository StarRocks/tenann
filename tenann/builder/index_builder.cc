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

#include "tenann/builder/index_builder.h"

#include "index_builder.h"
#include "tenann/common/logging.h"
#include "tenann/util/runtime_profile_macros.h"

namespace tenann {

IndexBuilder::IndexBuilder(const IndexMeta& meta) : index_meta_(meta){};

IndexBuilder::~IndexBuilder() {}

IndexBuilder& IndexBuilder::SetBuildOptions(const json& options) {
  T_LOG_IF(ERROR, is_opened()) << "all confuration actions must be called before index being opened";
  build_options_ = options;
  return *this;
}

IndexBuilder& IndexBuilder::SetIndexWriter(IndexWriterRef writer) {
  T_LOG_IF(ERROR, is_opened()) << "all confuration actions must be called before index being opened";
  index_writer_ = writer;
  return *this;
}

IndexBuilder& IndexBuilder::SetIndexCache(IndexCache* cache) {
  T_LOG_IF(ERROR, is_opened()) << "all confuration actions must be called before index being opened";
  T_CHECK_NOTNULL(cache);
  index_cache_ = cache;
  return *this;
}

IndexBuilder& IndexBuilder::EnableCustomRowId() {
  T_LOG_IF(ERROR, is_opened()) << "all confuration actions must be called before index being opened";
  use_custom_row_id_ = true;
  return *this;
}

IndexBuilder& IndexBuilder::EnableProfile() {
  T_LOG_IF(ERROR, is_opened()) << "all confuration actions must be called before index being opened";
  profile_ = std::make_unique<RuntimeProfile>("IndexBuilderProfile");
  PrepareProfile();
  return *this;
}

IndexBuilder& IndexBuilder::DisableProfile() {
  T_LOG_IF(ERROR, is_opened()) << "all confuration actions must be called before index being opened";
  profile_ = nullptr;
  return *this;
}

const IndexMeta& IndexBuilder::index_meta() const { return index_meta_; }

IndexRef IndexBuilder::index_ref() const { return index_ref_; }

IndexWriter* IndexBuilder::index_writer() { return index_writer_.get(); }

const IndexWriter* IndexBuilder::index_writer() const { return index_writer_.get(); }

IndexCache* IndexBuilder::index_cache() { return index_cache_; }

const IndexCache* IndexBuilder::index_cache() const { return index_cache_; }

RuntimeProfile* IndexBuilder::profile() { return profile_.get(); }

}  // namespace tenann