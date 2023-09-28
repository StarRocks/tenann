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

#include <type_traits>

#include "tenann/index/parameters.h"
#include "tenann/store/index_meta.h"

namespace tenann {

#define GET_OPTIONAL_PARAM_TO(meta, target, Section, parameter_name)                 \
  if (auto parameter_name =                                                          \
          (meta)                                                                     \
              .GetOptional##Section##Param<                                          \
                  std::remove_reference_t<decltype(target)>::parameter_name##_type>( \
                  std::remove_reference_t<decltype(target)>::parameter_name##_key);  \
      parameter_name.has_value()) {                                                  \
    (target).parameter_name = parameter_name.value();                                \
  }

#define GET_REQUIRED_PARAM_TO(meta, target, Section, parameter_name)             \
  (target).parameter_name =                                                      \
      (meta)                                                                     \
          .GetRequired##Section##Param<                                          \
              std::remove_reference_t<decltype(target)>::parameter_name##_type>( \
              std::remove_reference_t<decltype(target)>::parameter_name##_key);

#define GET_OPTIONAL_COMMON_PARAM_TO(meta, target, parameter_name) \
  GET_OPTIONAL_PARAM_TO(meta, target, Common, parameter_name)
#define GET_OPTIONAL_INDEX_PARAM_TO(meta, target, parameter_name) \
  GET_OPTIONAL_PARAM_TO(meta, target, Index, parameter_name)
#define GET_OPTIONAL_SEARCH_PARAM_TO(meta, target, parameter_name) \
  GET_OPTIONAL_PARAM_TO(meta, target, Search, parameter_name)
#define GET_OPTIONAL_EXTRA_PARAM_TO(meta, target, parameter_name) \
  GET_OPTIONAL_PARAM_TO(meta, target, Extra, parameter_name)

#define GET_REQUIRED_COMMON_PARAM_TO(meta, target, parameter_name) \
  GET_REQUIRED_PARAM_TO(meta, target, Common, parameter_name)
#define GET_REQUIRED_INDEX_PARAM_TO(meta, target, parameter_name) \
  GET_REQUIRED_PARAM_TO(meta, target, Index, parameter_name)
#define GET_REQUIRED_SEARCH_PARAM_TO(meta, target, parameter_name) \
  GET_REQUIRED_PARAM_TO(meta, target, Search, parameter_name)
#define GET_REQUIRED_EXTRA_PARAM_TO(meta, target, parameter_name) \
  GET_REQUIRED_PARAM_TO(meta, target, Extra, parameter_name)

inline void DeserializeParameters(const IndexMeta& meta, VectorIndexCommonParams* out_params) {
  GET_REQUIRED_COMMON_PARAM_TO(meta, *out_params, dim);
  GET_REQUIRED_COMMON_PARAM_TO(meta, *out_params, metric_type);
  GET_OPTIONAL_COMMON_PARAM_TO(meta, *out_params, is_vector_normed);
}

inline void DeserializeParameters(const IndexMeta& meta, FaissHnswIndexParams* out_params) {
  GET_OPTIONAL_INDEX_PARAM_TO(meta, *out_params, M);
  GET_OPTIONAL_INDEX_PARAM_TO(meta, *out_params, efConstruction);
}

inline void DeserializeParameters(const IndexMeta& meta, FaissHnswSearchParams* out_params) {
  GET_OPTIONAL_INDEX_PARAM_TO(meta, *out_params, efSearch);
  GET_OPTIONAL_INDEX_PARAM_TO(meta, *out_params, check_relative_distance);
}

inline void DeserializeParameters(const IndexMeta& meta, FaissIvfPqIndexParams* out_params) {
  GET_OPTIONAL_INDEX_PARAM_TO(meta, *out_params, nlists);
  GET_OPTIONAL_INDEX_PARAM_TO(meta, *out_params, M);
  GET_OPTIONAL_INDEX_PARAM_TO(meta, *out_params, nbits);
}

inline void DeserializeParameters(const IndexMeta& meta, FaissIvfPqSearchParams* out_params) {
  GET_OPTIONAL_INDEX_PARAM_TO(meta, *out_params, nprobe);
  GET_OPTIONAL_INDEX_PARAM_TO(meta, *out_params, max_codes);
  GET_OPTIONAL_INDEX_PARAM_TO(meta, *out_params, scan_table_threshold);
  GET_OPTIONAL_INDEX_PARAM_TO(meta, *out_params, polysemous_ht);
}

}  // namespace tenann