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

#include <limits.h>
#include <stddef.h>

#include <string>

#include "tenann/common/error.h"
#include "tenann/common/logging.h"

#define DEFINE_PARAM(type, key, default_value)                                  \
  using key##_type = type;                                                      \
  static constexpr const char* key##_key = #key;                                \
  static constexpr const type key##_default = static_cast<type>(default_value); \
  type key = static_cast<type>(default_value);

#define ASSERT_PARAM_IN_RANGE(param, min, max)                                      \
  if ((param) < (min) || (param) > (max)) {                                         \
    T_LOG(ERROR) << #param << " should in range [" << (min) << "," << (max) << "]"; \
  }

/// These two macros serve as document purposes to help users and developers understand a parameter
/// is required or not. They share the exactly same implementation,
#define DEFINE_OPTIONAL_PARAM(type, key, default_value) DEFINE_PARAM(type, key, default_value)
#define DEFINE_REQUIRED_PRARM(type, key, default_value) DEFINE_PARAM(type, key, default_value)

namespace tenann {

struct VectorIndexCommonParams {
  DEFINE_REQUIRED_PRARM(int, dim, 0);
  DEFINE_REQUIRED_PRARM(int, metric_type, 0);
  DEFINE_OPTIONAL_PARAM(bool, is_vector_normed, false);

  void Validate() {
    ASSERT_PARAM_IN_RANGE(dim, 1, 65536);
    ASSERT_PARAM_IN_RANGE(metric_type, 0, 4);
  }
};

struct VectorIndexExtraParams {
  std::string comments = "";

  void Validate() {}
};

/** Parameters for Faiss IVF-PQ */
struct FaissIvfPqIndexParams {
  DEFINE_OPTIONAL_PARAM(size_t, nlist, 16);
  DEFINE_OPTIONAL_PARAM(size_t, M, 2);
  DEFINE_OPTIONAL_PARAM(size_t, nbits, 8);

  void Validate() {
    ASSERT_PARAM_IN_RANGE(nlist, 1, INT_MAX);
    ASSERT_PARAM_IN_RANGE(M, 1, INT_MAX);
    ASSERT_PARAM_IN_RANGE(nbits, 8, 32);
  }
};

struct FaissIvfPqSearchParams {
  DEFINE_OPTIONAL_PARAM(size_t, nprobe, 1);
  DEFINE_OPTIONAL_PARAM(size_t, max_codes, 0);
  DEFINE_OPTIONAL_PARAM(size_t, scan_table_threshold, 0);
  DEFINE_OPTIONAL_PARAM(int, polysemous_ht, 0);
  DEFINE_OPTIONAL_PARAM(float, range_search_confidence, 0);

  void Validate() {
    ASSERT_PARAM_IN_RANGE(nprobe, 1, INT_MAX);
    ASSERT_PARAM_IN_RANGE(range_search_confidence, 0, 1);
  }
};

/**
 * Scalar/Product quantizer type for HNSW index.
 * Used by FaissHnswIndexParams::quantizer.
 *
 * - kFlat: no quantization, store raw float32 vectors (default, equivalent to HNSWFlat).
 * - kSQ4:  4-bit scalar quantization (ScalarQuantizer::QT_4bit).
 * - kSQ8:  8-bit scalar quantization (ScalarQuantizer::QT_8bit).
 * - kPQ:   Product quantization with m_pq sub-quantizers and nbits_pq bits each.
 */
enum class ScalarQuantizerType : int {
  kFlat = 0,
  kSQ4 = 1,
  kSQ8 = 2,
  kPQ = 3,
};

/** Parameters for faiss HNSW */
struct FaissHnswIndexParams {
  DEFINE_OPTIONAL_PARAM(int, M, 16);
  DEFINE_OPTIONAL_PARAM(int, efConstruction, 40);
  // Quantizer type (see ScalarQuantizerType). 0 (Flat) means no quantization.
  DEFINE_OPTIONAL_PARAM(int, quantizer, 0);
  // PQ parameters, only used when quantizer == kPQ.
  // m_pq: number of PQ sub-quantizers. Must divide `dim`.
  DEFINE_OPTIONAL_PARAM(int, m_pq, 0);
  // nbits_pq: number of bits per PQ sub-quantizer code, in [4, 16].
  DEFINE_OPTIONAL_PARAM(int, nbits_pq, 8);

  void Validate() {
    ASSERT_PARAM_IN_RANGE(M, 1, 65536);
    ASSERT_PARAM_IN_RANGE(efConstruction, 1, 65536);
    ASSERT_PARAM_IN_RANGE(quantizer, 0, 3);
    if (quantizer == static_cast<int>(ScalarQuantizerType::kPQ)) {
      ASSERT_PARAM_IN_RANGE(m_pq, 1, 1024);
      ASSERT_PARAM_IN_RANGE(nbits_pq, 4, 16);
    }
  }
};

struct FaissHnswSearchParams {
  DEFINE_OPTIONAL_PARAM(int, efSearch, 16);
  DEFINE_OPTIONAL_PARAM(bool, check_relative_distance, true);

  void Validate() { ASSERT_PARAM_IN_RANGE(efSearch, 1, INT_MAX); }
};

struct IndexWriterOptions {
  DEFINE_OPTIONAL_PARAM(bool, write_index_cache, false);
  std::string custom_cache_key = "";
  std::string cosine_backend = "l2";

  void Validate() {
    T_CHECK(cosine_backend == "l2" || cosine_backend == "inner_product")
        << "cosine_backend must be l2 or inner_product";
  }
};

struct IndexReaderOptions {
  DEFINE_OPTIONAL_PARAM(bool, cache_index_file, false);
  std::string custom_cache_key = "";
  DEFINE_OPTIONAL_PARAM(bool, force_read_and_overwrite_cache, false);
  DEFINE_OPTIONAL_PARAM(bool, cache_index_block, false);

  void Validate() {}
};

}  // namespace tenann
