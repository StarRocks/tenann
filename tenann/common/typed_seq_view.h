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
  const T* data;
  idx_t size;
};

template <typename T>
struct TypedVlArraySeqView {
  TypedVlArraySeqView() : data(nullptr), offsets(nullptr), size(0) {}

  explicit TypedVlArraySeqView(const VlArraySeqView& seq_view) {
    constexpr const PrimitiveType expected_type = RuntimePrimitiveType<T>::primitive_type;
    T_CHECK_EQ(seq_view.elem_type, expected_type);
    T_DCHECK_NE(seq_view.elem_type, PrimitiveType::kUnknownType);

    data = reinterpret_cast<const T*>(seq_view.data);
    offsets = seq_view.offsets;
    size = seq_view.size;
  }

  TypedVlArraySeqView(const T* data, const uint32_t* offsets, uint32_t size)
      : data(data), offsets(offsets), size(size) {}

  class iterator {
   public:
    // iterator traits
    using difference_type = idx_t;
    using value_type = TypedSlice<T>;
    using pointer = TypedSlice<T>*;
    using reference = TypedSlice<T>&;
    using iterator_category = std::forward_iterator_tag;

    iterator(const TypedVlArraySeqView* typed_view, idx_t i) : typed_view_(typed_view), i_(i) {}

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
    const TypedVlArraySeqView* typed_view_;
    idx_t i_;
  };

  iterator begin() const { return iterator(this, 0); }
  iterator end() const { return iterator(this, size); }

  const T* data;
  const uint32_t* offsets;
  uint32_t size;
};

template <typename T>
struct TypedArraySeqView {
  TypedArraySeqView() : data(nullptr), dim(0), size(0) {}

  explicit TypedArraySeqView(const ArraySeqView& seq_view) {
    constexpr const PrimitiveType expected_type = RuntimePrimitiveType<T>::primitive_type;
    T_CHECK_EQ(seq_view.elem_type, expected_type);
    T_DCHECK_NE(seq_view.elem_type, PrimitiveType::kUnknownType);

    data = reinterpret_cast<const T*>(seq_view.data);
    dim = seq_view.dim;
    size = seq_view.size;
  }

  TypedArraySeqView(T* data, uint32_t dim, uint32_t size) : data(data), dim(dim), size(size) {}

  class iterator {
   public:
    // iterator traits
    using difference_type = idx_t;
    using value_type = TypedSlice<T>;
    using pointer = TypedSlice<T>*;
    using reference = TypedSlice<T>&;
    using iterator_category = std::forward_iterator_tag;

    iterator(const TypedArraySeqView* typed_view, idx_t i) : typed_view_(typed_view), i_(i) {}

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
      return TypedSlice<T>{.data = typed_view_->data + typed_view_->dim * i_,
                           .size = typed_view_->dim};
    }

   private:
    const TypedArraySeqView* typed_view_;
    idx_t i_;
  };

  iterator begin() const { return iterator(this, 0); }
  iterator end() const { return iterator(this, size); }

  const T* data;
  uint32_t dim;
  uint32_t size;
};

template <typename T>
class TypedSliceIterator {
 public:
  T_FORBID_DEFAULT_CTOR(TypedSliceIterator);

  TypedSliceIterator(const ArraySeqView& view)
      : view_{.array_seq_view = view}, is_vl_array_(false) {
    constexpr const PrimitiveType expected_type = RuntimePrimitiveType<T>::primitive_type;
    T_CHECK_EQ(view.elem_type, expected_type);
    T_DCHECK_NE(view.elem_type, PrimitiveType::kUnknownType);
  }

  TypedSliceIterator(const VlArraySeqView& view)
      : view_{.vl_array_seq_view = view}, is_vl_array_(true) {
    constexpr const PrimitiveType expected_type = RuntimePrimitiveType<T>::primitive_type;
    T_CHECK_EQ(view.elem_type, expected_type);
    T_DCHECK_NE(view.elem_type, PrimitiveType::kUnknownType);
  }

  /**
   * @brief Iterate over an ArraySeqView or VlArraySeqView.
   *
   * An example:
   * @code
   *   ArraySeqView view = ...;
   *   auto iter = TypedSliceIterator<float>(view_);
   *   iter.ForEach([](const float* slice_data, idx_t slice_length) {
   *       ...
   *   })
   * @endcode
   *
   *
   * @tparam F
   * @param lambda A function of type `void f(const T*, idx_t)`.
   */
  template <typename F>
  void ForEach(F&& lambda) const {
    if (is_vl_array_) {
      const T* data = reinterpret_cast<const T*>(view_.vl_array_seq_view.data);
      const uint32_t* offsets = view_.vl_array_seq_view.offsets;
      idx_t size = view_.vl_array_seq_view.size;

      for (idx_t i = 0; i < size; i++) {
        const T* slice_data = data + offsets[i];
        idx_t slice_length = offsets[i + 1] - offsets[i];
        lambda(i, slice_data, slice_length);
      }
    } else {
      const T* data = reinterpret_cast<const T*>(view_.array_seq_view.data);
      idx_t dim = view_.array_seq_view.dim;
      idx_t size = view_.array_seq_view.size;
      const T* end = data + size * dim;

      idx_t i = 0;
      while (data < end) {
        lambda(i, data, dim);
        data += dim;
        i += 1;
      }
    }
  }

  const T* data() const {
    if (is_vl_array_) {
      return reinterpret_cast<const T*>(view_.vl_array_seq_view.data);
    } else {
      return reinterpret_cast<const T*>(view_.array_seq_view.data);
    }
  }

  idx_t size() const {
    if (is_vl_array_) {
      return static_cast<idx_t>(view_.vl_array_seq_view.size);
    } else {
      return static_cast<idx_t>(view_.array_seq_view.size);
    }
  }

 private:
  union {
    ArraySeqView array_seq_view;
    VlArraySeqView vl_array_seq_view;
  } view_;
  bool is_vl_array_;
  bool is_null_;
};

}  // namespace tenann