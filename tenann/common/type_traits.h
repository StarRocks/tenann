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

#include "tenann/common/primitive_type.h"

namespace tenann {

typedef int64_t idx_t;

template <typename T>
struct RuntimePrimitiveType {
  static const PrimitiveType primitive_type = PrimitiveType::kUnknownType;
};

template <>
struct RuntimePrimitiveType<bool> {
  static const PrimitiveType primitive_type = PrimitiveType::kBoolType;
};

template <>
struct RuntimePrimitiveType<int8_t> {
  static const PrimitiveType primitive_type = PrimitiveType::kInt8Type;
};

template <>
struct RuntimePrimitiveType<int16_t> {
  static const PrimitiveType primitive_type = PrimitiveType::kInt16Type;
};

template <>
struct RuntimePrimitiveType<int32_t> {
  static const PrimitiveType primitive_type = PrimitiveType::kInt32Type;
};

template <>
struct RuntimePrimitiveType<int64_t> {
  static const PrimitiveType primitive_type = PrimitiveType::kInt64Type;
};

template <>
struct RuntimePrimitiveType<uint8_t> {
  static const PrimitiveType primitive_type = PrimitiveType::kUInt8Type;
};

template <>
struct RuntimePrimitiveType<uint16_t> {
  static const PrimitiveType primitive_type = PrimitiveType::kUInt16Type;
};

template <>
struct RuntimePrimitiveType<uint32_t> {
  static const PrimitiveType primitive_type = PrimitiveType::kUInt32Type;
};

template <>
struct RuntimePrimitiveType<uint64_t> {
  static const PrimitiveType primitive_type = PrimitiveType::kUInt64Type;
};

template <>
struct RuntimePrimitiveType<float> {
  static const PrimitiveType primitive_type = PrimitiveType::kFloatType;
};

template <>
struct RuntimePrimitiveType<double> {
  static const PrimitiveType primitive_type = PrimitiveType::kDoubleType;
};

}  // namespace tenann
