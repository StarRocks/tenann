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

#include <faiss/impl/IDSelector.h>

#include <memory>
#include <vector>

#include "tenann/searcher/id_filter.h"

namespace tenann {

class IdFilterAdapter : public faiss::IDSelector {
 public:
  IdFilterAdapter(const IdFilter* id_filter, const std::vector<int64_t>* id_map = nullptr)
      : id_filter_(id_filter), id_map_(id_map) {}

  bool is_member(int64_t id) const override {
    if (id_filter_ == nullptr) {
      return true;
    }

    if (id_map_) {
      return id_filter_->IsMember((*id_map_)[id]);
    }
    return id_filter_->IsMember(id);
  }

 private:
  const IdFilter* id_filter_;
  const std::vector<int64_t>* id_map_;
};

class IdFilterAdapterFactory {
 public:
  static std::shared_ptr<IdFilterAdapter> CreateIdFilterAdapter(
      const IdFilter* id_filter, const std::vector<int64_t>* id_map = nullptr) {
    return std::make_shared<IdFilterAdapter>(id_filter, id_map);
  }
};

struct IDSelectorRangeAdapter : faiss::IDSelectorRange {
  IDSelectorRangeAdapter(idx_t imin, idx_t imax, bool assume_sorted = false)
      : IDSelectorRange(imin, imax, assume_sorted) {}
};

struct IDSelectorArrayAdapter : faiss::IDSelectorArray {
  IDSelectorArrayAdapter(size_t n, const idx_t* ids) : IDSelectorArray(n, ids) {}
};

struct IDSelectorBatchAdapter : faiss::IDSelectorBatch {
  IDSelectorBatchAdapter(size_t n, const idx_t* indices) : IDSelectorBatch(n, indices) {}
};

struct IDSelectorBitmapAdapter : faiss::IDSelectorBitmap {
  IDSelectorBitmapAdapter(size_t n, const uint8_t* bitmap) : IDSelectorBitmap(n, bitmap) {}
};
}  // namespace tenann