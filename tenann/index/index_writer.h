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

#include "tenann/common/json.hpp"
#include "tenann/index/index.h"

namespace tenann {

class IndexWriter {
 public:
  explicit IndexWriter(const IndexMeta& meta) : index_meta_(meta){};
  virtual ~IndexWriter();

  T_FORBID_DEFAULT_CTOR(IndexWriter);
  T_FORBID_COPY_AND_ASSIGN(IndexWriter);
  T_FORBID_MOVE(IndexWriter);

  // Write index file
  virtual void WriteIndex(IndexRef index, const std::string& path) = 0;

  nlohmann::json& conf();
  const nlohmann::json& conf() const;

  /** Getters */
  const IndexMeta& index_meta() const;

 protected:
  // @TODO: consider using a shared_ptr to save index meta,
  // otherwise there are too many meta copies.
  /* meta */
  IndexMeta index_meta_;
  /* write options */
  nlohmann::json conf_;
};

using IndexWriterRef = std::shared_ptr<IndexWriter>;

}  // namespace tenann