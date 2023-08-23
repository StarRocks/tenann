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

#include "tenann/common/macros.h"
#include "tenann/index/index_reader.h"
#include "tenann/store/index_cache.h"
#include "tenann/store/index_meta.h"

namespace tenann {

template <typename SearcherImpl>
class Searcher {
 public:
  virtual ~Searcher() = default;
  TNN_FORBID_COPY_AND_ASSIGN(Searcher);
  TNN_FORBID_MOVE(Searcher);

  SearcherImpl& ReadIndex(const std::string& path, bool force_read_and_flush_cache = false) {
    assert(index_reader_ != nullptr);
    index_ = index_reader_->ReadIndex(path);
    return static_cast<SearcherImpl&>(*this);
  };

  /// Set single search parameter.
  SearcherImpl& SetSearchParamItem(const std::string& key, const json& value) {
    search_params_[key] = value;
    this->SearchParamItemChangeHook(key, value);
    return static_cast<SearcherImpl&>(*this);
  };

  /// Set all search parameters.
  SearcherImpl& SetSearchParams(const nlohmann::json& params) {
    search_params_ = params;
    this->SearchParamsChangeHook(params);
    return static_cast<SearcherImpl&>(*this);
  }

  /* setters and getters */

  SearcherImpl& SetIndexReader(IndexReader* reader) {
    index_reader_ = reader;
    return static_cast<SearcherImpl&>(*this);
  }

  SearcherImpl& SetIndexCache(IndexCache* cache) {
    index_cache_ = cache;
    return static_cast<SearcherImpl&>(*this);
  }

  IndexRef index() const { return index_; }

  const IndexReader* index_reader() const { return index_reader_; }

  IndexReader* index_reader() { return index_reader_; }

  const IndexCache* index_cache() const { return index_cache_; }

  IndexCache* index_cache() { return index_cache_; }

  bool is_index_loaded() const { return is_index_loaded_; }

 protected:
  virtual void SearchParamItemChangeHook(const std::string& key, const json& value) = 0;

  virtual void SearchParamsChangeHook(const json& value) = 0;

  IndexMeta index_meta_;
  IndexRef index_;
  bool is_index_loaded_;
  json search_params_;

  /* reader and cache */
  IndexReader* index_reader_;
  IndexCache* index_cache_;
};

}  // namespace tenann
