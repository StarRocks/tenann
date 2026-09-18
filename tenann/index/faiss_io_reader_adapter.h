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
#include "faiss/impl/maybe_owned_vector.h"
#include "tenann/store/index_file_reader.h"
#include "tenann/util/stop_watch.h"

namespace tenann {

/// A faiss::IOReader implementation that delegates all reads to an
/// IndexFileReader, allowing FAISS to read from remote file systems.
class FaissIOReaderAdapter : public faiss::IOReader {
 public:
  /// The reader's owner, expressed as the type FAISS keeps. Holding it here is what
  /// releases the block when the last view into it is gone -- and not before.
  class FaissMemoryOwner : public faiss::MaybeOwnedVectorOwner {
   public:
    explicit FaissMemoryOwner(std::shared_ptr<MemoryOwner> owner) : owner_(std::move(owner)) {}

   private:
    std::shared_ptr<MemoryOwner> owner_;
  };

  explicit FaissIOReaderAdapter(IndexFileReaderPtr reader)
      : reader_(std::move(reader)), bytes_read_(0), io_time_ns_(0) {
    name = reader_->filename();
  }

  /// Called by FAISS IO macros (READ1, READVECTOR, etc.).
  /// Semantics match fread(): returns number of complete items read.
  size_t operator()(void* ptr, size_t size, size_t nitems) override {
    int64_t total = static_cast<int64_t>(size) * static_cast<int64_t>(nitems);
    MonotonicStopWatch sw;
    sw.start();
    int64_t n = reader_->Read(ptr, total);
    sw.stop();
    io_time_ns_ += sw.elapsed_time();
    if (n <= 0) return 0;
    bytes_read_ += static_cast<size_t>(n);
    return static_cast<size_t>(n) / size;
  }

  /// Let the underlying reader own the allocation of the next large array. The two
  /// owner types are kept apart deliberately: FAISS' lives here, TenANN's is what the
  /// reader interface exposes, and this is the only place they meet.
  void* allocate_for_read(size_t bytes,
                          std::shared_ptr<faiss::MaybeOwnedVectorOwner>* owner) override {
    std::shared_ptr<MemoryOwner> from_reader;
    void* address = reader_->AllocateForRead(bytes, &from_reader);
    if (address != nullptr) {
      *owner = std::make_shared<FaissMemoryOwner>(std::move(from_reader));
    }
    return address;
  }

  /// Returns the total number of bytes read through this IOReader so far.
  /// Used to determine the file offset at which inverted lists data starts.
  size_t bytes_read() const { return bytes_read_; }

  /// Returns cumulative I/O time in nanoseconds.
  uint64_t io_time_ns() const { return io_time_ns_; }

  IndexFileReaderPtr reader_;

 private:
  size_t bytes_read_;
  uint64_t io_time_ns_;
};

}  // namespace tenann
