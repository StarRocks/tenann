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
/// heap. While the graph is built it can serve distances from an 8-bit copy of the rows. Graph
/// construction is bound by the latency of reading candidate vectors, and a quarter of the bytes
/// roughly halves it; the stored vectors, the written index and search stay float. It serializes
/// as a plain IndexFlat.
class HnswBuildStorage : public faiss::IndexFlat {
 public:
  /// |unit_range|: rows lie in [-1, 1] (normalized), so the 8-bit range needs no pass over them.
  HnswBuildStorage(faiss::idx_t d, faiss::MetricType metric, MemoryAccountantPtr accountant,
                   bool int8_build, bool unit_range);
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
  faiss::DistanceComputer* get_distance_computer() const override;

  /// Drops the 8-bit copy. Distances are float from then on.
  void FinishBuild();

  bool int8_active() const { return int8_on_; }

 private:
  float* xb() const { return reinterpret_cast<float*>(rows_); }
  void Reserve(size_t rows);
  void SyncView();
  // 8-bit codes for rows [from, to); widens the range (re-encoding earlier rows) when a row falls
  // outside it.
  void Encode(size_t from, size_t to);
  void EncodeRows(size_t from, size_t to);

  MemoryAccountantPtr accountant_;
  uint8_t* rows_ = nullptr;  // float rows, capacity_ of them
  uint8_t* int8_ = nullptr;  // 8-bit codes, capacity_ of them while int8_on_
  bool int8_on_ = false;
  size_t capacity_ = 0;
  faiss::idx_t num_staged_ = 0;
  bool unit_range_ = false;
  bool has_range_ = false;
  float lo_ = 0;
  float scale_ = 1;  // code = round((x - lo_) * scale_)
};

namespace hnsw_build_internal {
/// Squared L2 over 8-bit codes with the kernel of |level| ("scalar", "avx2", "avx512"), one
/// candidate and four at once. Returns false when this CPU lacks the level. For tests.
bool L2U8WithLevel(const char* level, const uint8_t* q, const uint8_t* const* y, size_t d,
                   int64_t* one, int64_t* four);
}  // namespace hnsw_build_internal

}  // namespace tenann
