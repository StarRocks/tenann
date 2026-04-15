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

#include "tenann/builder/faiss_index_builder_with_buffer.h"
#include "tenann/index/parameters.h"

namespace faiss {
class IndexHNSW;
}

namespace tenann {

/// Index builder for faiss HNSW. Inherits from FaissIndexBuilderWithBuffer so
/// rows added via Add() are buffered and then dispatched to faiss in a single
/// add(N, ...) call from Flush(). Batch-add lets faiss's hnsw_add_vertices
/// engage its OpenMP parallel region (guarded by n > 100), giving multi-core
/// speedup over per-row insertion. Peak buffer memory is bounded by
/// SetFlushThresholdRows().
///
/// HNSWFlat is the only supported variant for now: the underlying faiss index
/// reports `is_trained == true` at construction so the train phase in Flush()
/// is a no-op. A future PR will plug quantized HNSW (SQ/PQ) into the train
/// phase and the partial-flush path.
class FaissHnswIndexBuilder final : public FaissIndexBuilderWithBuffer {
 public:
  using FaissIndexBuilderWithBuffer::FaissIndexBuilderWithBuffer;
  virtual ~FaissHnswIndexBuilder();

  T_FORBID_COPY_AND_ASSIGN(FaissHnswIndexBuilder);
  T_FORBID_MOVE(FaissHnswIndexBuilder);

 protected:
  IndexRef InitIndex() override;

 private:
  FaissHnswIndexParams index_params_;
  FaissHnswSearchParams search_params_;
};

}  // namespace tenann
