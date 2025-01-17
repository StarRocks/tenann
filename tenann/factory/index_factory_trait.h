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

#include <memory>

#include "tenann/builder/faiss_hnsw_index_builder.h"
#include "tenann/builder/faiss_ivf_pq_index_builder.h"
#include "tenann/builder/index_builder.h"
#include "tenann/common/error.h"
#include "tenann/index/index_ivfpq_reader.h"
#include "tenann/index/index_ivfpq_writer.h"
#include "tenann/index/faiss_index_reader.h"
#include "tenann/index/faiss_index_writer.h"
#include "tenann/index/index_reader.h"
#include "tenann/index/index_writer.h"
#include "tenann/store/index_meta.h"
#include "tenann/store/index_type.h"

namespace tenann {

#define CASE_ALL_INDEX_TYPE                                          \
  case kFaissHnsw: {                                                 \
    CASE_FN(kFaissHnsw);                                             \
    break;                                                           \
  }                                                                  \
  case kFaissIvfPq: {                                                \
    CASE_FN(kFaissIvfPq);                                            \
    break;                                                           \
  }                                                                  \
  default: {                                                         \
    throw Error(__FILE__, __LINE__, "using unsupported index type"); \
  }

/// Currently, we use this template class to create the IndexReader, IndexWriter, etc.,
/// for specific index types.
/// In the future, we plan to use the abstract factory pattern to dispatch index types
/// and create corresponding factories, which will help improve maintainability
template <IndexType type>
struct IndexFactoryTrait {
  [[noreturn]] static std::shared_ptr<IndexReader> CreateReaderFromMeta(const IndexMeta& meta) {
    throw Error(__FILE__, __LINE__, "method not implemented: CreateReaderFromMeta");
  };

  [[noreturn]] static std::shared_ptr<IndexWriter> CreateWriterFromMeta(const IndexMeta& meta) {
    throw Error(__FILE__, __LINE__, "method not implemented: CreateWriterFromMeta");
  };

  [[noreturn]] static std::shared_ptr<IndexBuilder> CreateBuilderFromMeta(const IndexMeta& meta) {
    throw Error(__FILE__, __LINE__, "method not implemented: CreateBuilderFromMeta");
  };
};

template <>
struct IndexFactoryTrait<kFaissHnsw> {
  static std::shared_ptr<IndexReader> CreateReaderFromMeta(const IndexMeta& meta) {
    return std::make_shared<FaissIndexReader>(meta);
  };

  static std::shared_ptr<IndexWriter> CreateWriterFromMeta(const IndexMeta& meta) {
    return std::make_shared<FaissIndexWriter>(meta);
  };

  static std::shared_ptr<IndexBuilder> CreateBuilderFromMeta(const IndexMeta& meta) {
    return std::make_shared<FaissHnswIndexBuilder>(meta);
  };
};

template <>
struct IndexFactoryTrait<kFaissIvfPq> {
  static std::shared_ptr<IndexReader> CreateReaderFromMeta(const IndexMeta& meta) {
    return std::make_shared<IndexIvfPqReader>(meta);
  };

  static std::shared_ptr<IndexWriter> CreateWriterFromMeta(const IndexMeta& meta) {
    return std::make_shared<IndexIvfPqWriter>(meta);
  };

  static std::shared_ptr<IndexBuilder> CreateBuilderFromMeta(const IndexMeta& meta) {
    return std::make_shared<FaissIvfPqIndexBuilder>(meta);
  };
};

}  // namespace tenann