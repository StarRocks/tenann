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
#include "tenann/index/index_reader.h"
#include "tenann/store/index_meta.h"
#include "tenann/factory/index_factory.h"

namespace tenann {

/**
 * @brief Base class for all searchers. Not thread-safe.
 *
 * @tparam ChildSearcher
 */
template <typename ChildSearcher>
class Searcher {
 public:
  explicit Searcher(const IndexMeta& meta) : index_meta_(meta) {
    index_reader_ = IndexFactory::CreateReaderFromMeta(meta);
    index_reader_->SetIndexCache(IndexCache::GetGlobalInstance());
  }
  virtual ~Searcher() = default;

  T_FORBID_DEFAULT_CTOR(Searcher);
  T_FORBID_COPY_AND_ASSIGN(Searcher);
  T_FORBID_MOVE(Searcher);

  ChildSearcher& ReadIndex(const std::string& path) {
    index_ref_ = index_reader_->ReadIndex(path);
    is_index_loaded_ = true;

    OnIndexLoaded();
    return static_cast<ChildSearcher&>(*this);
  };

  /// Set single search parameter.
  ChildSearcher& SetSearchParamItem(const std::string& key, const json& value) {
    this->OnSearchParamItemChange(key, value);
    return static_cast<ChildSearcher&>(*this);
  };

  /// Set all search parameters.
  ChildSearcher& SetSearchParams(const json& params) {
    this->OnSearchParamsChange(params);
    return static_cast<ChildSearcher&>(*this);
  }

  /* setters and getters */
  IndexRef index_ref() const { return index_ref_; }

  const IndexReader* index_reader() const { return index_reader_.get(); }

  IndexReader* index_reader() { return index_reader_.get(); }

  bool is_index_loaded() const { return is_index_loaded_; }

 protected:
  virtual void OnSearchParamItemChange(const std::string& key, const json& value) = 0;

  virtual void OnSearchParamsChange(const json& value) = 0;

  virtual void OnIndexLoaded(){};

  IndexMeta index_meta_;
  IndexRef index_ref_;
  bool is_index_loaded_;

  /* reader */
  IndexReaderRef index_reader_;
};

}  // namespace tenann
