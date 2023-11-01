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

namespace tenann {

class AnnFilterAdapterBase {
 public:
  virtual ~AnnFilterAdapterBase() {}
  virtual bool isMember(idx_t id) const = 0;
  virtual faiss::IDSelector* getIDSelector() const = 0;
};

class AnnFilterRangeAdapter : public AnnFilterAdapterBase {
 public:
  AnnFilterRangeAdapter(idx_t min_id, idx_t max_id, bool assume_sorted)
      : min_id_(min_id), max_id_(max_id), assume_sorted_(assume_sorted) {
    id_selector_ = std::make_unique<faiss::IDSelectorRange>(min_id_, max_id_, assume_sorted_);
  }

  bool isMember(idx_t id) const override { return id_selector_->is_member(id); }

  faiss::IDSelector* getIDSelector() const override { return id_selector_.get(); }

 private:
  idx_t min_id_;
  idx_t max_id_;
  bool assume_sorted_;
  std::unique_ptr<faiss::IDSelectorRange> id_selector_;
};

class AnnFilterArrayAdapter : public AnnFilterAdapterBase {
 public:
  AnnFilterArrayAdapter(const idx_t* ids, size_t num_ids) {
    id_array_.assign(ids, ids + num_ids);
    id_selector_ = std::make_unique<faiss::IDSelectorArray>(id_array_.size(), id_array_.data());
  }

  bool isMember(idx_t id) const override { return id_selector_->is_member(id); }

  faiss::IDSelector* getIDSelector() const override { return id_selector_.get(); }

 private:
  std::vector<idx_t> id_array_;
  std::unique_ptr<faiss::IDSelectorArray> id_selector_;
};

class AnnFilterBatchAdapter : public AnnFilterAdapterBase {
 public:
  AnnFilterBatchAdapter(const idx_t* ids, size_t num_ids) {
    id_selector_ = std::make_unique<faiss::IDSelectorBatch>(num_ids, ids);
  }

  bool isMember(idx_t id) const override { return id_selector_->is_member(id); }

  faiss::IDSelector* getIDSelector() const override { return id_selector_.get(); }

 private:
  std::unique_ptr<faiss::IDSelectorBatch> id_selector_;
};

class AnnFilterBitmapAdapter : public AnnFilterAdapterBase {
 public:
  AnnFilterBitmapAdapter(const uint8_t* bitmap, size_t bitmap_size) {
    bitmap_.assign(bitmap, bitmap + bitmap_size);
    id_selector_ = std::make_unique<faiss::IDSelectorBitmap>(bitmap_.size() * 8, bitmap_.data());
  }

  bool isMember(idx_t id) const override { return id_selector_->is_member(id); }

  faiss::IDSelector* getIDSelector() const override { return id_selector_.get(); }

 private:
  std::vector<uint8_t> bitmap_;
  std::unique_ptr<faiss::IDSelectorBitmap> id_selector_;
};
}  // namespace tenann