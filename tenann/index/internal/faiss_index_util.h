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

#include "faiss/IndexIDMap.h"
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

  if (common_params.metric_type == MetricType::kCosineSimilarity &&
      !common_params.is_vector_normed) {
    oss << "L2Norm,";
  }

  if (use_custom_rowid) {
    oss << "IDMap,";
  }

  oss << "HNSW";
  oss << index_params.M;
  return oss.str();
}

inline const faiss::IndexHNSW* UnpackHnsw(const faiss::Index* index,
                                          const VectorIndexCommonParams& common_params,
                                          const FaissHnswIndexParams& index_params,
                                          bool use_custom_rowid = false) {
  const faiss::Index* sub_index = index;

  if (common_params.metric_type == MetricType::kCosineSimilarity &&
      !common_params.is_vector_normed) {
    auto target = FAISS_DOWN_CAST(faiss::IndexPreTransform, sub_index);
    sub_index = target->index;
  }

  if (use_custom_rowid) {
    auto target = FAISS_DOWN_CAST(faiss::IndexIDMap, sub_index);
    sub_index = target->index;
  }

  auto hnsw = FAISS_DOWN_CAST(faiss::IndexHNSW, sub_index);

  return hnsw;
}

inline faiss::IndexHNSW* UnpackHnswMutable(faiss::Index* index,
                                           const VectorIndexCommonParams& common_params,
                                           const FaissHnswIndexParams& index_params,
                                           bool use_custom_rowid = false) {
  return const_cast<faiss::IndexHNSW*>(
      UnpackHnsw(index, common_params, index_params, use_custom_rowid));
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
  oss << "IVF" << index_params.nlists << ",";
  oss << "PQ" << index_params.M << "x" << index_params.nbits;

  return oss.str();
}

inline const faiss::IndexIVFPQ* UnpackIvfPq(const faiss::Index* index,
                                            const VectorIndexCommonParams& common_params,
                                            const FaissIvfPqIndexParams& index_params) {
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
                                             const VectorIndexCommonParams& common_params,
                                             const FaissIvfPqIndexParams& index_params) {
  return const_cast<faiss::IndexIVFPQ*>(UnpackIvfPq(index, common_params, index_params));
}

}  // namespace faiss_util

}  // namespace tenann