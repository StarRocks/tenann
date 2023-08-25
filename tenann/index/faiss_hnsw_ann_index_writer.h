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

#include "tenann/common/json.hpp"
#include "tenann/index/index_writer.h"

namespace tenann {

class FaissHnswAnnIndexWriter : public IndexWriter {
 public:
  FaissHnswAnnIndexWriter(const IndexMeta& meta);
  FaissHnswAnnIndexWriter() = delete;
  virtual ~FaissHnswAnnIndexWriter();
  T_FORBID_COPY_AND_ASSIGN(FaissHnswAnnIndexWriter);
  T_FORBID_MOVE(FaissHnswAnnIndexWriter);

  // Write index file
  void WriteIndex(IndexRef index, const std::string& path) override;
};

}  // namespace tenann