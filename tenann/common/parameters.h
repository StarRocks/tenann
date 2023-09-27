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

#define DEFINE_PARAM(type, key, default_value)                 \
  using key##_type = type;                                     \
  static constexpr const char* key##_key = #key;               \
  static constexpr const type key##_default = (default_value); \
  type key = (default_value);

#define DEFINE_SEARCH_PARAM(type, key, default_value)     \
  using search_##key##_type = type;                       \
  static constexpr const char* search_##key##_key = #key; \
  static constexpr const type search_##key##_default = (default_value);

namespace tenann {

struct IndexParamsFaissIvfPq {
  DEFINE_PARAM(size_t, nlists, 16);
  DEFINE_PARAM(size_t, M, 2);
  DEFINE_PARAM(size_t, nbits, 8);
};

struct SearchParamsFaissInvfPq {
  DEFINE_PARAM(size_t, nprobe, 1);
};

struct IndexParamsFaissHnsw {
  DEFINE_PARAM(int, M, 16);
  DEFINE_PARAM(int, efConstruction, 40);
};

struct SearchParamsFaissHnsw {
  DEFINE_PARAM(int, efSearch, 1);
  DEFINE_PARAM(bool, check_relative_distance, true);
};

}  // namespace tenann
