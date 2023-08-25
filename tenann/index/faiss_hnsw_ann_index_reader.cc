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

#include <faiss/index_io.h>
#include <faiss/IndexHNSW.h>

#include "tenann/index/faiss_hnsw_ann_index_reader.h"

namespace tenann {

FaissHnswAnnIndexReader::FaissHnswAnnIndexReader(const IndexMeta& meta) {
  index_meta_ = meta;
}

FaissHnswAnnIndexReader::~FaissHnswAnnIndexReader() = default;

IndexRef FaissHnswAnnIndexReader::ReadIndex(const std::string& path) {
  auto index_hnsw = std::unique_ptr<faiss::IndexHNSW>(dynamic_cast<faiss::IndexHNSW*>(faiss::read_index(path.c_str(), faiss::IO_FLAG_MMAP)));
  return std::make_shared<Index>(index_hnsw.release(), IndexType::kFaissHnsw, [](void* index) {delete static_cast<faiss::IndexHNSW*>(index);});
}

}  // namespace tenann