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

namespace tenann {

enum PrimitiveType {
  kUnknownType = 0, /* 0 */
  kBoolType,        /* 1 */
  kInt8Type,        /* 2 */
  kInt16Type,       /* 3 */
  kInt32Type,       /* 4 */
  kInt64Type,       /* 5 */
  kUInt8Type,       /* 6 */
  kUInt16Type,      /* 7 */
  kUInt32Type,      /* 8 */
  kUInt64Type,      /* 9 */
  kFloatType,       /* 10 */
  kDoubleType       /* 11 */
};

}