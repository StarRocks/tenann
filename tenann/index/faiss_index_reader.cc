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

#include "tenann/index/faiss_index_reader.h"

#include "faiss/Index.h"
#include "faiss/impl/FaissException.h"
#include "faiss/index_io.h"
#include "tenann/common/logging.h"

namespace tenann {

FaissIndexReader::~FaissIndexReader() = default;

IndexRef FaissIndexReader::ReadIndexFile(const std::string& path) {
  try {
    auto faiss_index =
        std::unique_ptr<faiss::Index>(faiss::read_index(path.c_str(), faiss::IO_FLAG_MMAP));
    return std::make_shared<Index>(faiss_index.release(),  //
                                   (IndexType)index_meta_.index_type(),
                                   [](void* index) { delete static_cast<faiss::Index*>(index); });
  } catch (faiss::FaissException& e) {
    T_LOG(ERROR) << e.what();
  }
}

}  // namespace tenann