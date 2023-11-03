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

#include <sstream>
#include <string>
#include <utility>

#include "faiss/IndexHNSW.h"
#include "faiss/IndexIDMap.h"
#include "faiss/IndexIVFPQ.h"
#include "faiss/IndexPreTransform.h"
#include "tenann/common/error.h"
#include "tenann/index/index.h"
#include "tenann/index/parameters.h"
#include "tenann/store/index_type.h"

namespace tenann {

namespace faiss_util {

template <typename TargetIndexType>
inline const TargetIndexType* faiss_down_cast(const faiss::Index* index, const char* target_name) {
  auto target_index = dynamic_cast<const TargetIndexType*>(index);
  T_CHECK(target_index != nullptr)
      << "cannot unpack faiss `" << target_name
      << "` from the type-erased pointer, either "
         "the given index meta or the in-memory index structure is broken";
  return target_index;
}

#define FAISS_DOWN_CAST(target_index_type, index) \
  faiss_down_cast<target_index_type>((index), #target_index_type)

/************************************************************
 * Faiss HNSW index
 ************************************************************/

inline std::string GetHnswRepr(const VectorIndexCommonParams& common_params,
                               const FaissHnswIndexParams& index_params,
                               bool use_custom_rowid = false) {
  std::ostringstream oss;

  if (use_custom_rowid) {
    oss << "IDMap,";
  }

  if (common_params.metric_type == MetricType::kCosineSimilarity &&
      !common_params.is_vector_normed) {
    oss << "L2Norm,";
  }

  oss << "HNSW";
  oss << index_params.M;
  return oss.str();
}

inline std::tuple<const faiss::IndexIDMap*, const faiss::IndexPreTransform*,
                  const faiss::IndexHNSW*>
UnpackHnsw(const faiss::Index* index, const VectorIndexCommonParams& common_params) {
  const faiss::Index* sub_index = index;

  const faiss::IndexIDMap* id_map = dynamic_cast<const faiss::IndexIDMap*>(index);
  const faiss::IndexPreTransform* transform = nullptr;

  if (id_map != nullptr) {
    sub_index = id_map->index;
  }

  if (common_params.metric_type == MetricType::kCosineSimilarity &&
      !common_params.is_vector_normed) {
    transform = FAISS_DOWN_CAST(faiss::IndexPreTransform, sub_index);
    sub_index = transform->index;
  }

  auto hnsw = FAISS_DOWN_CAST(faiss::IndexHNSW, sub_index);

  return std::make_tuple(id_map, transform, hnsw);
}

inline std::tuple<faiss::IndexIDMap*, faiss::IndexPreTransform*, faiss::IndexHNSW*>
UnpackHnswMutable(faiss::Index* index, const VectorIndexCommonParams& common_params) {
  auto [id_map, transform, hnsw] = UnpackHnsw(index, common_params);
  return std::make_tuple(const_cast<faiss::IndexIDMap*>(id_map),
                         const_cast<faiss::IndexPreTransform*>(transform),
                         const_cast<faiss::IndexHNSW*>(hnsw));
}

/************************************************************
 * Faiss IVF-PQ index
 ************************************************************/

inline std::string GetIvfPqRepr(const VectorIndexCommonParams& common_params,
                                const FaissIvfPqIndexParams& index_params) {
  std::ostringstream oss;

  if (common_params.metric_type == MetricType::kCosineSimilarity &&
      !common_params.is_vector_normed) {
    oss << "L2Norm,";
  }
  oss << "IVF" << index_params.nlist << ",";
  oss << "PQ" << index_params.M << "x" << index_params.nbits;

  return oss.str();
}

inline const faiss::IndexIVFPQ* UnpackIvfPq(const faiss::Index* index,
                                            const VectorIndexCommonParams& common_params) {
  const faiss::Index* sub_index = index;

  if (common_params.metric_type == MetricType::kCosineSimilarity &&
      !common_params.is_vector_normed) {
    auto target = FAISS_DOWN_CAST(faiss::IndexPreTransform, sub_index);
    sub_index = target->index;
  }

  auto ivfpq = FAISS_DOWN_CAST(faiss::IndexIVFPQ, sub_index);

  return ivfpq;
}

inline faiss::IndexIVFPQ* UnpackIvfPqMutable(faiss::Index* index,
                                             const VectorIndexCommonParams& common_params) {
  return const_cast<faiss::IndexIVFPQ*>(UnpackIvfPq(index, common_params));
}

}  // namespace faiss_util

}  // namespace tenann