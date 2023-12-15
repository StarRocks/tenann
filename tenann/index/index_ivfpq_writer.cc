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

#include "tenann/index/index_ivfpq_writer.h"

#include "faiss/Index.h"
#include "faiss/impl/FaissAssert.h"
#include "faiss/impl/FaissException.h"
#include "faiss/impl/io.h"
#include "faiss/impl/io_macros.h"
#include "faiss/index_io.h"
#include "tenann/common/logging.h"
#include "tenann/index/internal/index_ivfpq.h"
#include "tenann/util/defer.h"

namespace tenann {

IndexIvfPqWriter::~IndexIvfPqWriter() = default;

void write_index_header(const faiss::Index* idx, faiss::IOWriter* f) {
  WRITE1(idx->d);
  WRITE1(idx->ntotal);
  faiss::Index::idx_t dummy = 1 << 20;
  WRITE1(dummy);
  WRITE1(dummy);
  WRITE1(idx->is_trained);
  WRITE1(idx->metric_type);
  if (idx->metric_type > 1) {
    WRITE1(idx->metric_arg);
  }
}

void IndexIvfPqWriter::WriteIndexFile(IndexRef index, const std::string& path) {
  // open the index file and close it automatically
  // when we leave the current scope through `Defer`
  auto file = fopen(path.c_str(), "wb");
  Defer defer([file]() {
    if (file != nullptr) fclose(file);
  });

  T_LOG_IF(ERROR, file == nullptr)
      << "could not open[" << path << "] for writing: " << strerror(errno);

  try {
    // write custom fields with faiss FileIOWriter and IO macros
    faiss::FileIOWriter writer(file);
    writer.name = path;
    // the name `f` is needed for faiss IO macros
    auto* f = &writer;
    const auto* faiss_index = static_cast<const faiss::Index*>(index->index_raw());

    if (const IndexIvfPq* index_ivfpq = dynamic_cast<const IndexIvfPq*>(faiss_index)) {
      faiss::write_index(index_ivfpq, f);
      // write range_search_confidence
      WRITE1(index_ivfpq->range_search_confidence);
      // write reconstruction errors
      size_t vec_size = index_ivfpq->reconstruction_errors.size();
      WRITE1(vec_size);
      for (const auto& sub_vec : index_ivfpq->reconstruction_errors) {
        WRITEVECTOR(sub_vec);
      }
    } else if (const faiss::IndexPreTransform* ixpt =
                   dynamic_cast<const faiss::IndexPreTransform*>(faiss_index)) {
      uint32_t h = faiss::fourcc("IxPT");
      WRITE1(h);
      write_index_header(ixpt, f);
      int nt = ixpt->chain.size();
      WRITE1(nt);
      for (int i = 0; i < nt; i++) {
        write_VectorTransform(ixpt->chain[i], f);
      }
      const IndexIvfPq* index_ivfpq = dynamic_cast<const IndexIvfPq*>(ixpt->index);
      faiss::write_index(index_ivfpq, f);
      WRITE1(index_ivfpq->range_search_confidence);
      // write reconstruction errors
      size_t vec_size = index_ivfpq->reconstruction_errors.size();
      WRITE1(vec_size);
      for (const auto& sub_vec : index_ivfpq->reconstruction_errors) {
        WRITEVECTOR(sub_vec);
      }
    } else {
      faiss::write_index(faiss_index, f);
      T_LOG(INFO) << "Unknow index to writer. using faiss::write_index()";
    }
  } catch (faiss::FaissException& e) {
    T_LOG(ERROR) << e.what();
  }
}

}  // namespace tenann