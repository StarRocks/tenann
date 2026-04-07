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

#include <memory>
#include <vector>

#include "tenann/common/type_traits.h"

namespace tenann {

class IDSelectorRangeAdapter;
class IDSelectorArrayAdapter;
class IDSelectorBatchAdapter;
class IDSelectorBitmapAdapter;

class IdFilter {
 public:
  virtual ~IdFilter() = 0;
  virtual bool IsMember(idx_t id) const = 0;
};

class RangeIdFilter : public IdFilter {
 public:
  /**
   * @brief Constructor, adapts IDSelectorRangeAdapter
   *
   * @param min_id Start of the ID range (inclusive)
   * @param max_id End of the ID range (exclusive)
   * @param assume_sorted Whether to assume the processed IDs are sorted
   *
   * If assume_sorted is true, the IDs being processed are assumed to be sorted.
   * In this case, the constructor will find the list index range where valid IDs are stored.
   * The returned range represents the start and end indices (exclusive) of valid IDs in the list.
   */
  RangeIdFilter(idx_t min_id, idx_t max_id, bool assume_sorted = false);
  ~RangeIdFilter() = default;

  bool IsMember(idx_t id) const override;

 private:
  std::shared_ptr<IDSelectorRangeAdapter> adapter_;
};

class ArrayIdFilter : public IdFilter {
 public:
  /**
   * @brief Constructor, adapts IDSelectorArrayAdapter
   *
   * @param ids Elements to store. The pointer can be released after construction completes
   * @param num_ids Number of IDs to store
   *
   * The constructor uses a simple array of elements.
   * In this case, is_member calls are less efficient, but some operations can use the IDs directly.
   */
  ArrayIdFilter(const idx_t* ids, size_t num_ids);
  ~ArrayIdFilter() = default;

  bool IsMember(idx_t id) const override;

 private:
  std::vector<idx_t> id_array_;
  std::shared_ptr<IDSelectorArrayAdapter> adapter_;
};

class BatchIdFilter : public IdFilter {
 public:
  /**
   * @brief Constructor, adapts IDSelectorBatchAdapter
   *
   * @param ids Elements to store. The pointer can be released after construction completes
   * @param num_ids Number of IDs to store
   *
   * The constructor uses IDs from a set (IDSelectorBatchAdapter).
   * Duplicate IDs do not affect performance when using bloom filters and sets.
   * The hash function used by the bloom filter and GCC's unordered_set is simply the
   * least significant bits of the ID. This works well for random IDs or IDs in consecutive
   * sequences, but will produce many hash collisions if the least significant bits are always the same.
   */
  BatchIdFilter(const idx_t* ids, size_t num_ids);
  ~BatchIdFilter() = default;

  bool IsMember(idx_t id) const override;

 private:
  std::shared_ptr<IDSelectorBatchAdapter> adapter_;
};

class BitmapIdFilter : public IdFilter {
 public:
  /**
   * @brief Constructor, adapts IDSelectorBitmapAdapter
   *
   * @param bitmap Binary mask array
   * @param bitmap_size Size of the binary mask array (ceil(n / 8))
   *
   * The constructor initializes the object using a binary mask.
   *
   * Note: Each element corresponds to one bit. The constructor uses a binary mask array of
   * size ceil(n / 8). An id is selected if and only if id / 8 < n and bit (i%8) of
   * bitmap[floor(i / 8)] is set to 1.
   */
  BitmapIdFilter(const uint8_t* bitmap, size_t bitmap_size);
  ~BitmapIdFilter() = default;

  bool IsMember(idx_t id) const override;

 private:
  std::shared_ptr<IDSelectorBitmapAdapter> adapter_;
};

}  // namespace tenann
