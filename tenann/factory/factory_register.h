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

#include "tenann/store/index_type.h"

namespace tenann {

/// !Do not include this header in your user code
#define CASE_ALL_INDEX_TYPE \
  case kFaissHnsw: {        \
    CASE_FN(kFaissHnsw);    \
    break;                  \
  }                         \
  case kFaissIvfPq: {       \
    CASE_FN(kFaissIvfPq);   \
    break;                  \
  }                         \
  case kFaissIvfFlat: {     \
    CASE_FN(kFaissIvfFlat); \
    break;                  \
  }                         \
  default:                  \
    throw "unknown index"  // @TODO: use exception instead

template <IndexType type>
struct FactoryDispatch {
  using IndexReaderClass = IndexReader;
  using IndexWriterClass = IndexReader;
  using IndexBuilderClass = IndexReader;
  using IndexStreamerClass = IndexReader;
  using IndexScannerClass = IndexReader;
};

template <>
struct FactoryDispatch<kFaissHnsw> {
  using IndexReaderClass = IndexReader;
  using IndexWriterClass = IndexReader;
  using IndexBuilderClass = IndexReader;
  using IndexStreamerClass = IndexReader;
  using IndexScannerClass = IndexReader;
};

}  // namespace tenann