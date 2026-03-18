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

#include <cstdint>
#include <memory>
#include <string>

namespace tenann {

/// Abstract file writer interface that allows TenANN to write index files
/// to arbitrary file systems (local, S3, HDFS, etc.).
///
/// Callers implement this interface to bridge their own file system
/// abstraction into TenANN's index writing path.
class IndexFileWriter {
 public:
  virtual ~IndexFileWriter() = default;

  /// Write |count| bytes from |data| to the file.
  /// Returns the number of bytes actually written, or -1 on error.
  virtual int64_t Write(const void* data, int64_t count) = 0;

  /// Flush buffered data to the underlying storage.
  virtual void Flush() = 0;

  /// Close the file. No further writes are allowed after this call.
  virtual void Close() = 0;

  /// Returns the file name / path (used for logging and identification).
  virtual const std::string& filename() const = 0;
};

using IndexFileWriterPtr = std::shared_ptr<IndexFileWriter>;

}  // namespace tenann
