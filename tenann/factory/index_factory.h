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

#include "tenann/builder/index_builder.h"
#include "tenann/common/logging.h"
#include "tenann/index/index_reader.h"
#include "tenann/index/index_writer.h"
#include "tenann/store/index_meta.h"

namespace tenann {

struct IndexFactory {
  static std::unique_ptr<IndexReader> CreateReaderFromMeta(const IndexMeta& meta);
  static std::unique_ptr<IndexWriter> CreateWriterFromMeta(const IndexMeta& meta);
  static std::unique_ptr<IndexBuilder> CreateBuilderFromMeta(const IndexMeta& meta);
};

}  // namespace tenann