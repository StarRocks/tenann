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

#include <cstddef>
#include <iterator>

#include "tenann/common/logging.h"
#include "tenann/common/seq_view.h"
#include "tenann/common/type_traits.h"

namespace tenann {

template <typename T>
struct TypedSlice {
  T* data;
  size_t size;
};

template <typename T>
struct TypedVlArraySeqView {
  TypedVlArraySeqView(const VlArraySeqView& seq_view) {
    constexpr const PrimitiveType expected_type = RuntimePrimitiveType<T>::primitive_type;
    T_CHECK_EQ(seq_view.elem_type, expected_type);
    T_DCHECK_NE(seq_view.elem_type, PrimitiveType::kUnknownType);

    data = reinterpret_cast<T*>(seq_view.data);
    offsets = seq_view.offsets;
    size = seq_view.size;
  }

  TypedVlArraySeqView(T* data, uint32_t* offsets, uint32_t size)
      : data(data), offsets(offsets), size(size) {}

  class iterator {
   public:
    // iterator traits
    using difference_type = size_t;
    using value_type = TypedSlice<T>;
    using pointer = TypedSlice<T>*;
    using reference = TypedSlice<T>&;
    using iterator_category = std::forward_iterator_tag;

    iterator(TypedVlArraySeqView* typed_view, size_t i) : typed_view_(typed_view), i_(i) {}

    iterator& operator++() {
      i_ += 1;
      return *this;
    }

    iterator operator++(int) {
      iterator ret_val = *this;
      ++(*this);
      return ret_val;
    }

    bool operator==(iterator other) const { return i_ == other.i_; }
    bool operator!=(iterator other) const { return i_ != other.i_; }

    TypedSlice<T> operator*() {
      return TypedSlice<T>{.data = typed_view_->data + typed_view_->offsets[i_],
                           .size = typed_view_->offsets[i_ + 1] - typed_view_->offsets[i_]};
    }

   private:
    TypedVlArraySeqView* typed_view_;
    size_t i_;
  };

  iterator begin() { return iterator(this, 0); }
  iterator end() { return iterator(this, size); }

  T* data;
  uint32_t* offsets;
  uint32_t size;
};

}  // namespace tenann