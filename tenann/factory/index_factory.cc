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

#include "tenann/factory/index_factory.h"

#include "tenann/factory/index_factory_trait.h"

namespace tenann {

// @TODO(petrizhang): use exceptions instead
std::unique_ptr<IndexReader> IndexFactory::CreateReaderFromMeta(const IndexMeta& meta) {
#define CASE_FN(TYPE) return IndexFactoryTrait<TYPE>::CreateReaderFromMeta(meta)

  std::unique_ptr<IndexReader> reader = nullptr;
  auto index_type = meta.index_type();
  switch (index_type) { CASE_ALL_INDEX_TYPE; }

#undef CASE_FN
}

std::unique_ptr<IndexWriter> IndexFactory::CreateWriterFromMeta(const IndexMeta& meta) {
#define CASE_FN(TYPE) return IndexFactoryTrait<TYPE>::CreateWriterFromMeta(meta)

  std::unique_ptr<IndexReader> reader = nullptr;
  auto index_type = meta.index_type();
  switch (index_type) { CASE_ALL_INDEX_TYPE; }

#undef CASE_FN
}

std::unique_ptr<IndexBuilder> IndexFactory::CreateBuilderFromMeta(const IndexMeta& meta) {
#define CASE_FN(TYPE) return IndexFactoryTrait<TYPE>::CreateBuilderFromMeta(meta)

  std::unique_ptr<IndexReader> reader = nullptr;
  auto index_type = meta.index_type();
  switch (index_type) { CASE_ALL_INDEX_TYPE; }

#undef CASE_FN
}

}  // namespace tenann