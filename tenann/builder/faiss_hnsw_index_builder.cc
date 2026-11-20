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

#include "tenann/builder/faiss_hnsw_index_builder.h"

#include <algorithm>
#include <limits>
#include <sstream>

#include "faiss/IndexHNSW.h"
#include "faiss/IndexIDMap.h"
#include "faiss/IndexPreTransform.h"
#include "faiss/VectorTransform.h"
#include "faiss/index_factory.h"
#include "faiss/utils/distances.h"
#include "faiss_hnsw_index_builder.h"
#include "tenann/common/logging.h"
#include "tenann/common/typed_seq_view.h"
#include "tenann/index/index.h"
#include "tenann/index/internal/faiss_index_util.h"
#include "tenann/index/internal/hnsw_build_storage.h"
#include "tenann/index/internal/metric_util.h"
#include "tenann/index/parameter_serde.h"
#include "tenann/util/runtime_profile.h"
#include "tenann/util/runtime_profile_macros.h"

namespace tenann {

FaissHnswIndexBuilder::FaissHnswIndexBuilder(const IndexMeta& meta)
    : FaissIndexBuilderWithBuffer(meta) {
  FetchParameters(meta, &index_params_);
  FetchParameters(meta, &search_params_);

  auto q = index_params_.quantizer;
  T_CHECK(q == static_cast<int>(ScalarQuantizerType::kFlat) ||
          q == static_cast<int>(ScalarQuantizerType::kSQ4) ||
          q == static_cast<int>(ScalarQuantizerType::kSQ8) ||
          q == static_cast<int>(ScalarQuantizerType::kPQ))
      << "invalid HNSW quantizer value: " << q;

  if (q == static_cast<int>(ScalarQuantizerType::kPQ)) {
    T_CHECK_GE(index_params_.nbits_pq, 4);
    T_CHECK_LE(index_params_.nbits_pq, 16);
    T_CHECK_GT(index_params_.m_pq, 0);
    T_CHECK_EQ(common_params_.dim % index_params_.m_pq, 0)
        << "HNSW+PQ requires dim (" << common_params_.dim << ") to be divisible by m_pq ("
        << index_params_.m_pq << ")";
    faiss_util::EstimateHnswPqSdcBytes(index_params_.m_pq, index_params_.nbits_pq);
  }
}

FaissHnswIndexBuilder::~FaissHnswIndexBuilder() {}

size_t FaissHnswIndexBuilder::GetMinTrainRows() const {
  switch (static_cast<ScalarQuantizerType>(index_params_.quantizer)) {
    case ScalarQuantizerType::kPQ:
      // faiss' recommendation: at least 100 training rows per centroid; the
      // PQ codebook has 2^nbits_pq centroids.
      T_CHECK_LT(static_cast<size_t>(index_params_.nbits_pq), std::numeric_limits<size_t>::digits);
      T_CHECK_LE(static_cast<size_t>(1) << index_params_.nbits_pq,
                 std::numeric_limits<size_t>::max() / 100);
      return (static_cast<size_t>(1) << index_params_.nbits_pq) * 100;
    case ScalarQuantizerType::kSQ4:
    case ScalarQuantizerType::kSQ8:
      return 1;
    case ScalarQuantizerType::kFlat:
    default:
      return 0;
  }
}

IndexRef FaissHnswIndexBuilder::InitIndex() {
  try {
    const int q = index_params_.quantizer;
    // create faiss index factory string
    auto factory_string =
        faiss_util::GetHnswRepr(common_params_, index_params_, use_custom_row_id_);
    const auto faiss_metric = ToFaissMetric(physical_metric_);

    // create faiss index
    auto index = std::unique_ptr<faiss::Index>(
        faiss::index_factory(common_params_.dim, factory_string.c_str(), faiss_metric));
    auto [_, transform, index_hnsw] =
        faiss_util::CheckAndUnpackHnswMutable(index.get(), &common_params_);
    T_CHECK_EQ(index_hnsw->metric_type, faiss_metric)
        << "faiss HNSW factory ignored the requested metric";
    T_CHECK_EQ(index_hnsw->storage->metric_type, faiss_metric)
        << "faiss HNSW storage ignored the requested metric";

    // set index parameters
    index_hnsw->hnsw.efConstruction = index_params_.efConstruction;
    // set default search paremeters
    index_hnsw->hnsw.efSearch = search_params_.efSearch;
    index_hnsw->hnsw.check_relative_distance = search_params_.check_relative_distance;

    VLOG(VERBOSE_DEBUG) << "efConstruction: " << index_hnsw->hnsw.efConstruction
                        << ", efSearch: " << index_hnsw->hnsw.efSearch
                        << ", check_relative_distance: "
                        << index_hnsw->hnsw.check_relative_distance;

    // Flat storage: swap in the build storage. It serializes as the IndexFlat it replaces. A
    // transform other than the L2Norm that GetHnswRepr adds keeps the generic path.
    build_storage_ = nullptr;
    normalize_rows_ = false;
    staged_ids_.clear();
    graph_broken_ = false;
    const bool l2norm_only =
        transform == nullptr ||
        (transform->chain.size() == 1 &&
         dynamic_cast<const faiss::NormalizationTransform*>(transform->chain[0]) != nullptr);
    if (q == static_cast<int>(ScalarQuantizerType::kFlat) && l2norm_only &&
        index_hnsw->own_fields && dynamic_cast<faiss::IndexFlat*>(index_hnsw->storage) != nullptr) {
      normalize_rows_ = transform != nullptr;
      const bool unit_range =
          normalize_rows_ || (common_params_.metric_type == MetricType::kCosineSimilarity &&
                              common_params_.is_vector_normed);
      auto storage =
          std::make_unique<HnswBuildStorage>(common_params_.dim, faiss_metric, memory_accountant_,
                                             index_writer_options_.hnsw_int8_build, unit_range);
      delete index_hnsw->storage;
      index_hnsw->storage = storage.release();
      build_storage_ = static_cast<HnswBuildStorage*>(index_hnsw->storage);
    }

    // create shared index ref
    return std::make_shared<Index>(index.release(),        //
                                   IndexType::kFaissHnsw,  //
                                   [](void* index) { delete static_cast<faiss::Index*>(index); });
  }
  CATCH_FAISS_ERROR
  CATCH_JSON_ERROR
}

IndexBuilder& FaissHnswIndexBuilder::Flush() {
  if (build_storage_ == nullptr) {
    return FaissIndexBuilderWithBuffer::Flush();
  }
  // Checked before the graph build changes the index, not after it as FaissIndexBuilder::Flush
  // does.
  T_LOG_IF(ERROR, !is_opened_) << "index builder has not been opened";
  T_LOG_IF(ERROR, is_closed_) << "index builder has already been closed";
  CheckGraphIntact();
  try {
    // The graph build; FaissIndexBuilder::Flush adds the write to the same timer.
    T_SCOPED_TIMER(flush_total_timer_);
    BuildStaged();
    build_storage_->FinishBuild();
    std::vector<idx_t>().swap(staged_ids_);
  }
  CATCH_FAISS_ERROR;
  return FaissIndexBuilder::Flush();
}

void FaissHnswIndexBuilder::AddRaw(const TypedSliceIterator<float>& input_row_iterator) {
  if (build_storage_ == nullptr) {
    return FaissIndexBuilderWithBuffer::AddRaw(input_row_iterator);
  }
  CheckGraphIntact();
  input_row_iterator.ForEach(
      [&](idx_t, const float* slice_data, idx_t) { StageRow(slice_data, nullptr); });
}

void FaissHnswIndexBuilder::AddWithRowIds(const TypedSliceIterator<float>& input_row_iterator,
                                          const idx_t* row_ids) {
  if (build_storage_ == nullptr) {
    return FaissIndexBuilderWithBuffer::AddWithRowIds(input_row_iterator, row_ids);
  }
  CheckGraphIntact();
  input_row_iterator.ForEach(
      [&](idx_t i, const float* slice_data, idx_t) { StageRow(slice_data, row_ids + i); });
}

void FaissHnswIndexBuilder::AddWithRowIdsAndNullFlags(
    const TypedSliceIterator<float>& input_row_iterator, const idx_t* row_ids,
    const uint8_t* null_flags) {
  if (build_storage_ == nullptr) {
    return FaissIndexBuilderWithBuffer::AddWithRowIdsAndNullFlags(input_row_iterator, row_ids,
                                                                  null_flags);
  }
  CheckGraphIntact();
  input_row_iterator.ForEach([&](idx_t i, const float* slice_data, idx_t) {
    if (null_flags[i] == 0) StageRow(slice_data, row_ids + i);
  });
}

void FaissHnswIndexBuilder::CheckGraphIntact() const {
  T_LOG_IF(ERROR, graph_broken_)
      << "an earlier HNSW graph build failed part way; the index can't be added to or written";
}

void FaissHnswIndexBuilder::StageRow(const float* row, const idx_t* row_id) {
  // Room for the id comes first: once the row is staged nothing may throw before its id is kept.
  if (row_id != nullptr && staged_ids_.size() == staged_ids_.capacity()) {
    staged_ids_.reserve(std::max<size_t>(1024, staged_ids_.capacity() * 2));
  }
  float* dst = build_storage_->StageRow();
  memcpy(dst, row, common_params_.dim * sizeof(float));
  if (row_id != nullptr) staged_ids_.push_back(*row_id);
  if (flush_threshold_rows_ > 0 &&
      static_cast<size_t>(build_storage_->num_staged()) >= flush_threshold_rows_) {
    BuildStaged();
  }
}

// Builds the graph for the staged rows. The rows are added to the HNSW directly, below the IDMap
// and the L2Norm transform, whose bookkeeping is then brought up to date; the transform's copy is
// never made.
void FaissHnswIndexBuilder::BuildStaged() {
  const idx_t n = build_storage_->num_staged();
  if (n == 0) return;
  auto [id_map, transform, index_hnsw] = faiss_util::UnpackHnswMutable(GetFaissIndex());
  T_CHECK(id_map == nullptr || static_cast<idx_t>(staged_ids_.size()) == n)
      << "rows of an index with custom row ids must be added with their row ids";
  float* rows = build_storage_->staged_data();
  if (normalize_rows_) {
    // What faiss's NormalizationTransform does to the rows it adds, over the whole batch at once.
    faiss::fvec_renorm_L2(common_params_.dim, n, rows);
  }
  // faiss has advanced the storage and the HNSW's ntotal before it inserts the vertices, and none
  // of this can be undone, so a throw from here on leaves the index broken for good.
  graph_broken_ = true;
  index_hnsw->add(n, rows);
  if (transform != nullptr) transform->ntotal = index_hnsw->ntotal;
  if (id_map != nullptr) {
    id_map->id_map.insert(id_map->id_map.end(), staged_ids_.begin(), staged_ids_.end());
    id_map->ntotal = index_hnsw->ntotal;
  }
  graph_broken_ = false;
  staged_ids_.clear();
}

}  // namespace tenann
