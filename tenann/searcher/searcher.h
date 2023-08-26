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

#include <cassert>

#include "tenann/common/macros.h"
#include "tenann/index/index_cache.h"
#include "tenann/index/index_reader.h"
#include "tenann/store/index_meta.h"

namespace tenann {

template <typename ChildSearcher>
class Searcher {
 public:
  explicit Searcher(const IndexMeta& meta) : index_meta_(meta){};
  virtual ~Searcher() = default;

  T_FORBID_DEFAULT_CTOR(Searcher);
  T_FORBID_COPY_AND_ASSIGN(Searcher);
  T_FORBID_MOVE(Searcher);

  ChildSearcher& ReadIndex(const std::string& path, bool read_index_cache = false,
                           bool use_custom_cache_key = false,
                           const std::string& custom_cache_key = "",
                           bool force_read_and_overwrite_cache = false) {
    if (read_index_cache) {
      auto cache_key = use_custom_cache_key ? custom_cache_key : path;
      assert(index_cache_ != nullptr);
      if (force_read_and_overwrite_cache) {
        ForceReadIndexAndOverwriteCache(path, cache_key);
      } else {
        auto found = index_cache_->Lookup(cache_key, &cache_handle_);
        if (found) {
          index_ref_ = cache_handle_.index_ref();
        } else {
          ForceReadIndexAndOverwriteCache(path, cache_key);
        }
      }
      is_index_loaded_ = true;
    } else {
      index_ref_ = index_reader_->ReadIndex(path);
      is_index_loaded_ = true;
    }

    return static_cast<ChildSearcher&>(*this);
  };

  /// Set single search parameter.
  ChildSearcher& SetSearchParamItem(const std::string& key, const json& value) {
    search_params_[key] = value;
    this->SearchParamItemChangeHook(key, value);
    return static_cast<ChildSearcher&>(*this);
  };

  /// Set all search parameters.
  ChildSearcher& SetSearchParams(const nlohmann::json& params) {
    search_params_ = params;
    this->SearchParamsChangeHook(params);
    return static_cast<ChildSearcher&>(*this);
  }

  /* setters and getters */

  ChildSearcher& SetIndexReader(IndexReader* reader) {
    index_reader_ = reader;
    return static_cast<ChildSearcher&>(*this);
  }

  ChildSearcher& SetIndexCache(IndexCache* cache) {
    index_cache_ = cache;
    return static_cast<ChildSearcher&>(*this);
  }

  IndexRef index_ref() const { return index_ref_; }

  const IndexReader* index_reader() const { return index_reader_; }

  IndexReader* index_reader() { return index_reader_; }

  const IndexCache* index_cache() const { return index_cache_; }

  IndexCache* index_cache() { return index_cache_; }

  bool is_index_loaded() const { return is_index_loaded_; }

 protected:
  void ForceReadIndexAndOverwriteCache(const std::string& path, const std::string& cache_key) {
    index_ref_ = index_reader_->ReadIndex(path);
    index_cache_->Insert(cache_key, index_ref_, &cache_handle_);
  }

  virtual void SearchParamItemChangeHook(const std::string& key, const json& value) = 0;

  virtual void SearchParamsChangeHook(const json& value) = 0;

  IndexMeta index_meta_;
  IndexRef index_ref_;
  bool is_index_loaded_;
  json search_params_;

  /* reader and cache */
  IndexReader* index_reader_;
  IndexCache* index_cache_;

  /**
   * @brief Use this handle to maintain a reference to the cache entry.
   *        Otherwise the cache entry may be cleaned when the reference count decreases to 1.
   *
   */
  IndexCacheHandle cache_handle_;
};

}  // namespace tenann
