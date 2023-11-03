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
#include "tenann/searcher/id_filter.h"

#include "tenann/searcher/internal/id_filter_adapter.h"

namespace tenann {

IDFilter::~IDFilter() = default;

// RangeIdFilter
RangeIdFilter::RangeIdFilter(idx_t min_id, idx_t max_id, bool assume_sorted) {
  adapter_ = std::make_shared<IDSelectorRangeAdapter>(min_id, max_id, assume_sorted);
}
bool RangeIdFilter::isMember(idx_t id) const { return adapter_->is_member(id); }

// ArrayIdFilter
ArrayIdFilter::ArrayIdFilter(const idx_t* ids, size_t num_ids) {
  id_array_.assign(ids, ids + num_ids);
  adapter_ = std::make_shared<IDSelectorArrayAdapter>(id_array_.size(), id_array_.data());
}
bool ArrayIdFilter::isMember(idx_t id) const { return adapter_->is_member(id); }

// BatchIdFilter
BatchIdFilter::BatchIdFilter(const idx_t* ids, size_t num_ids) {
  adapter_ = std::make_shared<IDSelectorBatchAdapter>(num_ids, ids);
}
bool BatchIdFilter::isMember(idx_t id) const { return adapter_->is_member(id); }

// BitmapIdFilter
BitmapIdFilter::BitmapIdFilter(const uint8_t* bitmap, size_t bitmap_size) {
  adapter_ = std::make_shared<IDSelectorBitmapAdapter>(bitmap_size, bitmap);
}
bool BitmapIdFilter::isMember(idx_t id) const { return adapter_->is_member(id); }

}  // namespace tenann