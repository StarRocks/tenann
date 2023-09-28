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

#include <stddef.h>

#include "faiss/IndexHNSW.h"
#include "faiss/IndexIVFPQ.h"

#define DEFINE_PARAM(type, key, default_value)                                  \
  using key##_type = type;                                                      \
  static constexpr const char* key##_key = #key;                                \
  static constexpr const type key##_default = static_cast<type>(default_value); \
  type key = static_cast<type>(default_value);

/// These two macros serve as document purposes to help users and developers understand a parameter
/// is required or not. They share the exactly same implementation,
#define DEFINE_OPTIONAL_PARAM(type, key, default_value) DEFINE_PARAM(type, key, default_value)
#define DEFINE_REQUIRED_PRARM(type, key, default_value) DEFINE_PARAM(type, key, default_value)

namespace tenann {

struct VectorIndexCommonParams {
  DEFINE_REQUIRED_PRARM(int, dim, 0);
  DEFINE_REQUIRED_PRARM(int, metric_type, 0);
  DEFINE_OPTIONAL_PARAM(bool, is_vector_normed, false);
};

/** Parameters for Faiss IVF-PQ */
struct FaissIvfPqIndexParams {
  DEFINE_OPTIONAL_PARAM(size_t, nlists, 16);
  DEFINE_OPTIONAL_PARAM(size_t, M, 2);
  DEFINE_OPTIONAL_PARAM(size_t, nbits, 8);
};

struct FaissIvfPqSearchParams {
  DEFINE_OPTIONAL_PARAM(size_t, nprobe, 1);
  DEFINE_OPTIONAL_PARAM(size_t, max_codes, 0);
  DEFINE_OPTIONAL_PARAM(size_t, scan_table_threshold, 0);
  DEFINE_OPTIONAL_PARAM(int, polysemous_ht, 0);
};

/** Parameters for faiss HSNW */
struct FaissHnswIndexParams {
  DEFINE_OPTIONAL_PARAM(int, M, 16);
  DEFINE_OPTIONAL_PARAM(int, efConstruction, 40);
};

struct FaissHnswSearchParams {
  DEFINE_OPTIONAL_PARAM(int, efSearch, 16);
  DEFINE_OPTIONAL_PARAM(bool, check_relative_distance, true);
};

}  // namespace tenann
