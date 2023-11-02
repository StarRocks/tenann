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

class IDFilter {
 public:
  virtual ~IDFilter() = 0;
  virtual bool isMember(idx_t id) const = 0;
};

class RangeIdFilter : public IDFilter {
 public:
  /**
   * @brief 构造函数，适配 IDSelectorRangeAdapter
   *
   * @param min_id 范围的起始 ID（包含）
   * @param max_id 范围的结束 ID（不包含）
   * @param assume_sorted 是否假设处理的 ID 已排序
   *
   * 如果 assume_sorted 为 true，则假设处理的 ID 是已排序的。
   * 在这种情况下，构造函数将查找存储有效 ID 的列表索引范围。
   * 返回的范围表示列表中有效 ID 存储的起始索引和结束索引（不包含结束索引）。
   */
  RangeIdFilter(idx_t min_id, idx_t max_id, bool assume_sorted = false);
  ~RangeIdFilter() = default;

  bool isMember(idx_t id) const override;

 private:
  std::unique_ptr<IDSelectorRangeAdapter> adapter_;
};

class ArrayIdFilter : public IDFilter {
  /**
   * @brief 构造函数，适配 IDSelectorArrayAdapter
   *
   * @param ids 要存储的元素。构造函数完成后，可以释放该指针
   * @param num_ids 要存储的 ID 数量
   *
   * 构造函数将使用简单的元素数组
   * 在这种情况下，is_member 调用的效率较低，但某些操作可以直接使用 ID。
   */
  ArrayIdFilter(const idx_t* ids, size_t num_ids);
  ~ArrayIdFilter() = default;

  bool isMember(idx_t id) const override;

 private:
  std::vector<idx_t> id_array_;
  std::unique_ptr<IDSelectorArrayAdapter> adapter_;
};

class BatchIdFilter : public IDFilter {
  /**
   * @brief 构造函数，适配IDSelectorBatchAdapter
   *
   * @param ids 要存储的元素。构造函数完成后，可以释放该指针
   * @param num_ids 要存储的 ID 数量
   *
   * 构造函数将使用集合中的 ID(IDSelectorBatchAdapter)。
   * 在使用布隆过滤器和集合时，重复的 ID 不会影响性能。
   * 布隆过滤器和 GCC 的 unordered_set 实现使用的哈希函数只是 ID 的最低有效位。
   * 这对于随机 ID 或连续序列中的 ID 是有效的，但如果最低有效位始终相同，则会产生许多哈希冲突。
   */
  BatchIdFilter(const idx_t* ids, size_t num_ids);
  ~BatchIdFilter() = default;

  bool isMember(idx_t id) const override;

 private:
  std::unique_ptr<IDSelectorBatchAdapter> adapter_;
};

class BitmapIdFilter : public IDFilter {
 public:
  /**
   * @brief 构造函数，适配 IDSelectorBitmapAdapter
   *
   * @param bitmap 二进制掩码数组
   * @param bitmap_size 二进制掩码数组的大小（ceil(n / 8)）
   *
   * 构造函数使用一个二进制掩码来初始化对象。
   *
   * 注意：每个元素对应一个位。构造函数使用一个二进制掩码数组，大小为 ceil(n / 8)。
   * 当且仅当 id / 8 < n 且 bitmap[floor(i / 8)] 的第 (i%8) 位为 1 时，该 id 会被选择。
   */
  BitmapIdFilter(const uint8_t* bitmap,
                 size_t bitmap_size);  // 构造函数，适配 IDSelectorBitmapAdapter
  ~BitmapIdFilter() = default;         // 析构函数

  bool isMember(idx_t id) const override;  // 公共接口函数

 private:
  std::unique_ptr<IDSelectorBitmapAdapter> adapter_;
};

}  // namespace tenann