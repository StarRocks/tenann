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
#include "tenann/common/macros.h"

namespace tenann {

/// Float storage of an HNSW Flat index under construction.
///
/// Rows are staged straight into the storage, so faiss's add() does not copy them: it only extends
/// the view over rows that are already in place. With an accountant the rows live in an anonymous
/// mapping charged to it, which grows with mremap instead of copying; without one they are on the
/// heap. It serializes as a plain IndexFlat.
///
/// The codes are a view over that block, so the faiss calls that resize codes in place
/// (remove_ids, merge_from, add_sa_codes) throw instead of asserting, and a copy of the index
/// (faiss::clone_index) must not outlive it. Read it back from its serialized form instead.
class HnswBuildStorage : public faiss::IndexFlat {
 public:
  HnswBuildStorage(faiss::idx_t d, faiss::MetricType metric, MemoryAccountantPtr accountant);
  ~HnswBuildStorage() override;

  T_FORBID_COPY_AND_ASSIGN(HnswBuildStorage);

  /// Space for the next staged row; valid until the next call. Staged rows join ntotal on add().
  float* StageRow();
  faiss::idx_t num_staged() const { return num_staged_; }
  float* staged_data() const { return xb() + ntotal * d; }

  /// Returns the block's spare capacity once no rows are staged, so a finished index holds only
  /// its rows. Later rows grow it again.
  void FinishBuild();
  size_t capacity() const { return capacity_; }

  /// Rows already staged are taken in place; any other input is copied in first.
  void add(faiss::idx_t n, const float* x) override;
  void reset() override;

  size_t remove_ids(const faiss::IDSelector& sel) override;
  void merge_from(faiss::Index& otherIndex, faiss::idx_t add_id = 0) override;
  void add_sa_codes(faiss::idx_t n, const uint8_t* x, const faiss::idx_t* xids) override;

 private:
  float* xb() const { return reinterpret_cast<float*>(rows_); }
  void Reserve(size_t rows);
  void SyncView();
  void FreeRows() noexcept;

  MemoryAccountantPtr accountant_;
  uint8_t* rows_ = nullptr;  // float rows, capacity_ of them
  size_t capacity_ = 0;
  faiss::idx_t num_staged_ = 0;
};

}  // namespace tenann
