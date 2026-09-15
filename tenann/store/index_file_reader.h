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

// Lets a consumer tell at compile time whether this bundle can read an index zero-copy.
// Without it, handing the reader a staged file only doubles the peak: the staged copy
// plus the one FAISS makes anyway.
#define TENANN_HAS_ZERO_COPY_READ 1

namespace tenann {

/// Abstract file reader interface that allows TenANN to read index files
/// from arbitrary file systems (local, S3, HDFS, etc.).
///
/// Callers implement this interface to bridge their own file system
/// abstraction into TenANN's index reading path.
class IndexFileReader {
 public:
  virtual ~IndexFileReader() = default;

  /// Sequential read: read up to |count| bytes from current position.
  /// Returns the number of bytes actually read, or -1 on error.
  virtual int64_t Read(void* data, int64_t count) = 0;

  /// Random read: read up to |count| bytes starting at |offset|.
  /// The current position is not affected. Concurrent calls to ReadAt must be
  /// safe and must not interfere with each other's offsets.
  /// Returns the number of bytes actually read, or -1 on error.
  virtual int64_t ReadAt(int64_t offset, void* data, int64_t count) = 0;

  /// Seek to the given absolute |position|.
  virtual void Seek(int64_t position) = 0;

  /// Returns the total file size in bytes.
  virtual int64_t GetSize() = 0;

  /// Returns the file name / path (used for cache key generation).
  virtual const std::string& filename() const = 0;

  /// Index bytes the reader already holds, offered to FAISS so it can point its arrays
  /// at them instead of allocating and copying its own. `owner` keeps the bytes alive and
  /// is captured by the index that views them, so they outlive every view into them.
  struct ZeroCopyBuffer {
    const uint8_t* data = nullptr;
    int64_t size = 0;
    std::shared_ptr<void> owner;
  };

  /// Optional zero-copy hand-off. Deserializing an index is otherwise pure resize+memcpy,
  /// which dominates the load of a large one.
  ///
  /// Precondition: the buffer must span the entire file, offset 0 to size. FAISS reaches
  /// the stream through one base pointer and a running cursor, so a per-range buffer
  /// would read the wrong bytes. Readers that stream keep the default and are unaffected.
  virtual ZeroCopyBuffer TryGetZeroCopyBuffer() { return ZeroCopyBuffer{}; }
};

using IndexFileReaderPtr = std::shared_ptr<IndexFileReader>;

}  // namespace tenann
