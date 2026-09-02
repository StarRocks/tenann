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

#include "tenann/builder/faiss_ivf_sq_index_builder.h"

#include "faiss/IndexFlat.h"
#include "faiss/IndexPreTransform.h"
#include "faiss/IndexScalarQuantizer.h"
#include "faiss/VectorTransform.h"
#include "tenann/common/logging.h"
#include "tenann/index/index.h"
#include "tenann/index/internal/metric_util.h"
#include "tenann/index/parameter_serde.h"

namespace tenann {

namespace {

faiss::ScalarQuantizer::QuantizerType GetFaissQuantizerType(size_t nbits) {
  switch (nbits) {
    case 4:
      return faiss::ScalarQuantizer::QT_4bit;
    case 8:
      return faiss::ScalarQuantizer::QT_8bit;
    default:
      T_CHECK(false) << "unsupported IVF-SQ nbits: " << nbits;
  }
  return faiss::ScalarQuantizer::QT_8bit;
}

}  // namespace

FaissIvfSqIndexBuilder::FaissIvfSqIndexBuilder(const IndexMeta& meta)
    : FaissIndexBuilderWithBuffer(meta) {
  FetchParameters(meta, &index_params_);
  FetchParameters(meta, &search_params_);
  T_CHECK(common_params_.metric_type == MetricType::kL2Distance ||
          common_params_.metric_type == MetricType::kCosineSimilarity ||
          common_params_.metric_type == MetricType::kInnerProduct)
      << "got unsupported metric; L2 distance, cosine similarity, and inner product are supported "
         "for IVF-SQ";
}

FaissIvfSqIndexBuilder::~FaissIvfSqIndexBuilder() = default;

IndexRef FaissIvfSqIndexBuilder::InitIndex() {
  try {
    const auto faiss_metric = ToFaissMetric(physical_metric_);

    auto quantizer = std::make_unique<faiss::IndexFlat>(common_params_.dim, faiss_metric);
    auto index_ivf_sq = std::make_unique<faiss::IndexIVFScalarQuantizer>(
        quantizer.release(), common_params_.dim, index_params_.nlist,
        GetFaissQuantizerType(index_params_.nbits), faiss_metric);
    index_ivf_sq->own_fields = true;
    index_ivf_sq->nprobe = search_params_.nprobe;
    index_ivf_sq->max_codes = search_params_.max_codes;
    index_ivf_sq->quantizer_trains_alone = 0;
    index_ivf_sq->cp.spherical = physical_metric_ == MetricType::kInnerProduct;

    VLOG(VERBOSE_DEBUG) << "nlist: " << index_ivf_sq->invlists->nlist
                        << ", nbits: " << index_params_.nbits;

    if (common_params_.metric_type == MetricType::kCosineSimilarity &&
        !common_params_.is_vector_normed) {
      auto index_pt = std::make_unique<faiss::IndexPreTransform>(index_ivf_sq.release());
      index_pt->own_fields = true;
      auto vector_transform =
          std::make_unique<faiss::NormalizationTransform>(common_params_.dim, 2.0);
      index_pt->prepend_transform(vector_transform.release());
      return std::make_shared<Index>(index_pt.release(), IndexType::kFaissIvfSq, [](void* index) {
        delete static_cast<faiss::IndexPreTransform*>(index);
      });
    }

    return std::make_shared<Index>(index_ivf_sq.release(), IndexType::kFaissIvfSq, [](void* index) {
      delete static_cast<faiss::IndexIVFScalarQuantizer*>(index);
    });
  }
  CATCH_FAISS_ERROR
  CATCH_JSON_ERROR
}

}  // namespace tenann
