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

#include "tenann/index/parameter_serde.h"
#include "tenann/store/index_meta.h"

namespace tenann {

constexpr const size_t kIvfPqMinRowsPerCluster = 39;

/**
 * @brief  Get minimum number of rows required by IndexIvfPq.
 *
 * @param meta Index meta.
 * @param min_rows_per_cluster Minimum data rows required by per cluster.
 *  If min_rows_per_cluster < 1, building ivfpq will throw an error.
 *  If 1 <= min_rows_per_cluster < 39, building ivfpq will trigger a warning.
 *  The best practice is to set min_rows_per_cluster to 39 or higher to avoid errors and
 *  warnings.
 * @return size_t
 */
inline size_t GetIvfPqMinRows(const IndexMeta& meta,
                              size_t min_rows_per_cluster = kIvfPqMinRowsPerCluster) {
  FaissIvfPqIndexParams params;
  FetchParameters(meta, &params);
  auto ivf_required_min_rows = min_rows_per_cluster * params.nlist;
  auto pq_required_min_rows = min_rows_per_cluster * (2 << params.nbits);
  auto min_rows =
      ivf_required_min_rows > pq_required_min_rows ? ivf_required_min_rows : pq_required_min_rows;
  return min_rows;
}

}  // namespace tenann