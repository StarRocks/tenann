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

#include "tenann/index/faiss_hnsw_index_writer.h"

#include <faiss/IndexHNSW.h>
#include <faiss/impl/io.h>
#include <faiss/index_io.h>

#include <fstream>

namespace tenann {

FaissHnswIndexWriter::~FaissHnswIndexWriter() = default;

void FaissHnswIndexWriter::WriteIndex(IndexRef index, const std::string& path) {
  auto faiss_index = static_cast<faiss::Index*>(index->index_raw());
  faiss::write_index(faiss_index, path.c_str());
}

}  // namespace tenann