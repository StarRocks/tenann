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

#include "tenann/index/index_ivfpq_reader.h"

#include "faiss/Index.h"
#include "faiss/IndexIVFPQR.h"
#include "faiss/impl/FaissAssert.h"
#include "faiss/impl/FaissException.h"
#include "faiss/impl/io.h"
#include "faiss/impl/io_macros.h"
#include "faiss/index_io.h"
#include "tenann/common/logging.h"
#include "tenann/index/internal/index_ivfpq.h"
#include "tenann/util/defer.h"

namespace faiss {

/*************************************************************
 * Copied from faiss/impl/index_read.cpp
 **************************************************************/

static void read_index_header(Index* idx, IOReader* f) {
  READ1(idx->d);
  READ1(idx->ntotal);
  Index::idx_t dummy;
  READ1(dummy);
  READ1(dummy);
  READ1(idx->is_trained);
  READ1(idx->metric_type);
  if (idx->metric_type > 1) {
    READ1(idx->metric_arg);
  }
  idx->verbose = false;
}

static void read_direct_map(DirectMap* dm, IOReader* f) {
  char maintain_direct_map;
  READ1(maintain_direct_map);
  dm->type = (DirectMap::Type)maintain_direct_map;
  READVECTOR(dm->array);
  if (dm->type == DirectMap::Hashtable) {
    using idx_t = Index::idx_t;
    std::vector<std::pair<idx_t, idx_t>> v;
    READVECTOR(v);
    std::unordered_map<idx_t, idx_t>& map = dm->hashtable;
    map.reserve(v.size());
    for (auto it : v) {
      map[it.first] = it.second;
    }
  }
}

static void read_ivf_header(IndexIVF* ivf, IOReader* f,
                            std::vector<std::vector<Index::idx_t>>* ids = nullptr) {
  read_index_header(ivf, f);
  READ1(ivf->nlist);
  READ1(ivf->nprobe);
  ivf->quantizer = read_index(f);
  ivf->own_fields = true;
  if (ids) {  // used in legacy "Iv" formats
    ids->resize(ivf->nlist);
    for (size_t i = 0; i < ivf->nlist; i++) READVECTOR((*ids)[i]);
  }
  read_direct_map(&ivf->direct_map, f);
}

// used for legacy formats
static ArrayInvertedLists* set_array_invlist(IndexIVF* ivf,
                                             std::vector<std::vector<Index::idx_t>>& ids) {
  ArrayInvertedLists* ail = new ArrayInvertedLists(ivf->nlist, ivf->code_size);
  std::swap(ail->ids, ids);
  ivf->invlists = ail;
  ivf->own_invlists = true;
  return ail;
}

static void read_InvertedLists(IndexIVF* ivf, IOReader* f, int io_flags) {
  InvertedLists* ils = read_InvertedLists(f, io_flags);
  if (ils) {
    FAISS_THROW_IF_NOT(ils->nlist == ivf->nlist);
    FAISS_THROW_IF_NOT(ils->code_size == InvertedLists::INVALID_CODE_SIZE ||
                       ils->code_size == ivf->code_size);
  }
  ivf->invlists = ils;
  ivf->own_invlists = true;
}

static void read_ProductQuantizer(ProductQuantizer* pq, IOReader* f) {
  READ1(pq->d);
  READ1(pq->M);
  READ1(pq->nbits);
  pq->set_derived_values();
  READVECTOR(pq->centroids);
}

/*************************************************************
 * Ported from faiss/impl/index_read.cpp
 **************************************************************/

static void read_ivfpq(IndexIVFPQ* ivpq, IOReader* f, uint32_t h, int io_flags) {
  bool legacy = h == fourcc("IvQR") || h == fourcc("IvPQ");

  std::vector<std::vector<Index::idx_t>> ids;
  read_ivf_header(ivpq, f, legacy ? &ids : nullptr);
  READ1(ivpq->by_residual);
  READ1(ivpq->code_size);
  read_ProductQuantizer(&ivpq->pq, f);

  if (legacy) {
    ArrayInvertedLists* ail = set_array_invlist(ivpq, ids);
    for (size_t i = 0; i < ail->nlist; i++) READVECTOR(ail->codes[i]);
  } else {
    read_InvertedLists(ivpq, f, io_flags);
  }

  if (ivpq->is_trained) {
    // precomputed table not stored. It is cheaper to recompute it.
    // precompute_table() may be disabled with a flag.
    ivpq->use_precomputed_table = 0;
    if (ivpq->by_residual) {
      if ((io_flags & IO_FLAG_SKIP_PRECOMPUTE_TABLE) == 0) {
        ivpq->precompute_table();
      }
    }
  }
}
}  // namespace faiss

namespace tenann {
using faiss::fourcc;
using faiss::fourcc_inv_printable;

// TODO: ignore this flag and use IndexCache
static constexpr const int IO_FLAG = faiss::IO_FLAG_READ_ONLY;

IndexIvfPqReader::~IndexIvfPqReader() = default;

IndexRef IndexIvfPqReader::ReadIndexFile(const std::string& path) {
  // open the index file and close it automatically
  // when we leave the current scope through `Defer`
  auto file = fopen(path.c_str(), "rb");
  Defer defer([file]() {
    if (file != nullptr) fclose(file);
  });

  T_LOG_IF(ERROR, file == nullptr)
      << "could not open [" << path << "] for reading: " << strerror(errno);

  try {
    // init an IOReader for index reading
    faiss::FileIOReader reader(file);
    reader.name = path;

    // the name `f` is needed for faiss IO macros
    auto* f = &reader;

    // read header
    uint32_t h;
    READ1(h);
    T_LOG_IF(ERROR, h != fourcc("IwPQ"))
        << "could not read ivfpq from file " << path << ": "
        << "expect magic number `IwPQ` but got " << fourcc_inv_printable(h);

    auto index_ivfpq = std::make_unique<IndexIvfPq>();
    // read faiss IndexIVFPQ
    faiss::read_ivfpq(index_ivfpq.get(), f, h, IO_FLAG);
    /* read custom fields */
    // read range_search_confidence
    READ1(index_ivfpq->range_search_confidence);
    // read reconstruction_errors
    size_t num_invlists;
    READ1(num_invlists);
    index_ivfpq->reconstruction_errors.resize(num_invlists);
    for (size_t i = 0; i < num_invlists; i++) {
      READVECTOR(index_ivfpq->reconstruction_errors[i]);
    }

    return std::make_shared<Index>(index_ivfpq.release(),  //
                                   IndexType::kFaissIvfPq,  //
                                   [](void* index) { delete static_cast<faiss::Index*>(index); });
  } catch (faiss::FaissException& e) {
    T_LOG(ERROR) << e.what();
  }
}

}  // namespace tenann