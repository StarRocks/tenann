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

#include "faiss/IndexFlat.h"
#include "tenann/builder/memory_accountant.h"

namespace tenann {

/// Float storage of an HNSW Flat index under construction.
///
/// Rows are staged straight into the storage, so faiss's add() does not copy them: it only extends
/// the view over rows that are already in place. With an accountant the rows live in an anonymous
/// mapping charged to it, which grows with mremap instead of copying; without one they are on the
/// heap. It serializes as a plain IndexFlat.
class HnswBuildStorage : public faiss::IndexFlat {
 public:
  HnswBuildStorage(faiss::idx_t d, faiss::MetricType metric, MemoryAccountantPtr accountant);
  ~HnswBuildStorage() override;

  HnswBuildStorage(const HnswBuildStorage&) = delete;
  HnswBuildStorage& operator=(const HnswBuildStorage&) = delete;

  /// Space for the next staged row; valid until the next call. Staged rows join ntotal on add().
  float* StageRow();
  faiss::idx_t num_staged() const { return num_staged_; }
  const float* staged_data() const { return xb() + ntotal * d; }

  /// Rows already staged are taken in place; any other input is copied in first.
  void add(faiss::idx_t n, const float* x) override;
  void reset() override;

 private:
  float* xb() const { return reinterpret_cast<float*>(rows_); }
  void Reserve(size_t rows);
  void SyncView();

  MemoryAccountantPtr accountant_;
  uint8_t* rows_ = nullptr;  // float rows, capacity_ of them
  size_t capacity_ = 0;
  faiss::idx_t num_staged_ = 0;
};

}  // namespace tenann
