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

#include <memory>

#include "tenann/common/json.h"
#include "tenann/index/index.h"
#include "tenann/index/index_cache.h"
#include "tenann/index/parameters.h"

namespace tenann {

class IndexWriter {
 public:
  explicit IndexWriter(const IndexMeta& meta);
  virtual ~IndexWriter();

  T_FORBID_DEFAULT_CTOR(IndexWriter);
  T_FORBID_COPY_AND_ASSIGN(IndexWriter);
  T_FORBID_MOVE(IndexWriter);

  // Write index cache
  void WriteIndex(IndexRef index, const std::string& path, bool memory_only);

  // Write index file
  virtual void WriteIndexFile(IndexRef index, const std::string& path) = 0;

  /** Setters */
  IndexWriter& SetIndexCache(IndexCache* cache);

  /** Getters */
  const IndexMeta& index_meta() const;
  IndexCache* index_cache();
  const IndexCache* index_cache() const;

 protected:
  // @TODO: consider using a shared_ptr to save index meta,
  // otherwise there are too many meta copies.
  /* meta */
  IndexMeta index_meta_;
  /* write options */
  WriteIndexOptions write_index_options_;
  /* cache */
  IndexCache* index_cache_ = nullptr;
};

using IndexWriterRef = std::shared_ptr<IndexWriter>;

}  // namespace tenann