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

#include <cstdint>

namespace tenann {

enum PrimitiveType {
  kInt32Type = 1,
  kInt64Type,
  kUInt32Type,
  kUint64Type,
  kFloatType,
  kDoubleType
};

/// View for a sequence of primitive c values.
struct PrimitiveSeqView {
  uint8_t* data;
  uint32_t size;
  PrimitiveType elem_type;
};

/// View for a sequence of variable-length arrays.
struct VlArraySeqView {
  uint8_t* data;
  uint32_t* offsets;
  uint32_t size;
  PrimitiveType elem_type;
};

/// View for a sequence of fixed-length arrays, i.e., matrix.
struct ArraySeqView {
  uint8_t* data;
  uint32_t dim;
  uint32_t size;
  PrimitiveType elem_type;
};

/// View for a sequence of strings.
struct StringSeqView {
  uint8_t* data;
  uint32_t* offsets;
  uint32_t size;
};

enum SeqViewType { kPrimitiveSeqView = 1, kArraySeqView, kVlArraySeqView, kStringSeqView };

// SeqView是所有类型的组合
struct SeqView {
  union {
    PrimitiveSeqView primitive_seq_view;
    ArraySeqView array_seq_view;
    VlArraySeqView vl_array_seq_view;
    StringSeqView string_seq_view;
  } seq_view;
  SeqViewType seq_view_type;

  /* implicit construction functions */

  SeqView(const PrimitiveSeqView& view)
      : seq_view{.primitive_seq_view = view}, seq_view_type(SeqViewType::kPrimitiveSeqView) {}

  SeqView(const ArraySeqView& view)
      : seq_view{.array_seq_view = view}, seq_view_type(SeqViewType::kArraySeqView) {}

  SeqView(const VlArraySeqView& view)
      : seq_view{.vl_array_seq_view = view}, seq_view_type(SeqViewType::kVlArraySeqView) {}

  SeqView(const StringSeqView& view)
      : seq_view{.string_seq_view = view}, seq_view_type(SeqViewType::kStringSeqView) {}
};

}  // namespace tenann