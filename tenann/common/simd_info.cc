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

#include "tenann/common/simd_info.h"

#include "faiss/utils/simd_levels.h"

namespace tenann {

std::string SimdLevelName() {
  // get_level_name() is defined inside faiss, so it reports the level faiss actually
  // selected. Do not substitute SIMDConfig::has_dynamic_dispatch() or any other
  // header-inlined constant here: FAISS_ENABLE_DD is private to the faiss target, so
  // such a constant evaluates against this translation unit's view, not faiss's.
  return faiss::SIMDConfig::get_level_name();
}

}  // namespace tenann
