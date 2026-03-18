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

#include <cstddef>
#include <cstdint>

#include "faiss/impl/io.h"
#include "tenann/store/index_file_writer.h"

namespace tenann {

/// A faiss::IOWriter implementation that delegates all writes to an
/// IndexFileWriter, allowing FAISS to write to remote file systems.
class FaissIOWriterAdapter : public faiss::IOWriter {
 public:
  explicit FaissIOWriterAdapter(IndexFileWriterPtr writer) : writer_(std::move(writer)) {
    name = writer_->filename();
  }

  /// Called by FAISS IO macros (WRITE1, WRITEVECTOR, etc.).
  /// Semantics match fwrite(): returns number of complete items written.
  size_t operator()(const void* ptr, size_t size, size_t nitems) override {
    int64_t total = static_cast<int64_t>(size) * static_cast<int64_t>(nitems);
    int64_t n = writer_->Write(ptr, total);
    if (n <= 0) return 0;
    return static_cast<size_t>(n) / size;
  }

 private:
  IndexFileWriterPtr writer_;
};

}  // namespace tenann
