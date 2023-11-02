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

#include "tenann/index/custom_ivfpq_writer.h"

#include "faiss/Index.h"
#include "faiss/impl/FaissAssert.h"
#include "faiss/impl/FaissException.h"
#include "faiss/impl/io.h"
#include "faiss/impl/io_macros.h"
#include "faiss/index_io.h"
#include "tenann/common/logging.h"
#include "tenann/index/internal/custom_ivfpq.h"
#include "tenann/util/defer.h"

namespace tenann {

CustomIvfPqWriter::~CustomIvfPqWriter() = default;

void CustomIvfPqWriter::WriteIndex(IndexRef index, const std::string& path) {
  // open the index file and close it automatically
  // when we leave the current scope through `Defer`
  auto file = fopen(path.c_str(), "wb");
  Defer defer([file]() {
    if (file != nullptr) fclose(file);
  });

  T_LOG_IF(ERROR, file == nullptr)
      << "could not open" << path << " for writing: " << strerror(errno);

  try {
    // write faiss index
    const auto* faiss_index = static_cast<const faiss::Index*>(index->index_raw());
    faiss::write_index(faiss_index, file);

    // write custom fields with faiss FileIOWriter and IO macros
    faiss::FileIOWriter writer(file);
    writer.name = path;

    // the name `f` is needed for faiss IO macros
    auto* f = &writer;

    // get the raw CustomIvfPq pointer
    const auto* custom_ivfpq = static_cast<const CustomIvfPq*>(index->index_raw());

    // write range_search_confidence
    WRITE1(custom_ivfpq->range_search_confidence);
    // write reconstruction errors
    size_t vec_size = custom_ivfpq->reconstruction_errors.size();
    WRITE1(vec_size);
    for (const auto& sub_vec : custom_ivfpq->reconstruction_errors) {
      WRITEVECTOR(sub_vec);
    }
  } catch (faiss::FaissException& e) {
    T_LOG(ERROR) << e.what();
  }
}

}  // namespace tenann