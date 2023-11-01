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
#include "ann_filter.h"

#include "tenann/searcher/internal/ann_filter_adapter.h"

namespace tenann {

AnnFilter::AnnFilter(idx_t min_id, idx_t max_id, bool assume_sorted) {
  adapter = std::make_unique<AnnFilterRangeAdapter>(min_id, max_id, assume_sorted);
}

AnnFilter::AnnFilter(const idx_t* ids, size_t num_ids, bool use_bloom_and_set) {
  if (use_bloom_and_set) {
    adapter = std::make_unique<AnnFilterBatchAdapter>(ids, num_ids);
  } else {
    adapter = std::make_unique<AnnFilterArrayAdapter>(ids, num_ids);
  }
}

AnnFilter::AnnFilter(const uint8_t* bitmap, size_t bitmap_size) {
  adapter = std::make_unique<AnnFilterBitmapAdapter>(bitmap, bitmap_size);
}

AnnFilter::~AnnFilter() {}

bool AnnFilter::isMember(idx_t id) const { return adapter->isMember(id); }

}  // namespace tenann