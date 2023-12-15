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

#include <cstdio>
#include <cstdlib>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>

#include "faiss/Index.h"
#include "faiss/IndexIVFPQR.h"
#include "faiss/impl/FaissAssert.h"
#include "faiss/impl/FaissException.h"
#include "faiss/impl/io.h"
#include "faiss/impl/io_macros.h"
#include "faiss/index_io.h"
#include "faiss/utils/hamming.h"
#include "faiss/invlists/InvertedListsIOHook.h"
#include "faiss/MetaIndexes.h"
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

static void read_ArrayInvertedLists_sizes(IOReader* f, std::vector<size_t>& sizes) {
  uint32_t list_type;
  READ1(list_type);
  if (list_type == fourcc("full")) {
    size_t os = sizes.size();
    READVECTOR(sizes);
    FAISS_THROW_IF_NOT(os == sizes.size());
  } else if (list_type == fourcc("sprs")) {
    std::vector<size_t> idsizes;
    READVECTOR(idsizes);
    for (size_t j = 0; j < idsizes.size(); j += 2) {
      FAISS_THROW_IF_NOT(idsizes[j] < sizes.size());
      sizes[idsizes[j]] = idsizes[j + 1];
    }
  } else {
    FAISS_THROW_FMT(
            "list_type %ud (\"%s\") not recognized",
            list_type,
            fourcc_inv_printable(list_type).c_str());
  }
}

InvertedLists* read_InvertedLists_with_block_cache(IOReader* f, int io_flags, tenann::IndexCache* index_cache) {
  uint32_t h;
  READ1(h);
  if (h == fourcc("il00")) {
    fprintf(stderr,
            "read_InvertedLists:"
            " WARN! inverted lists not stored with IVF object\n");
    return nullptr;
  } else if (h == fourcc("ilar")) {
    size_t nlist, code_size;
    READ1(nlist);
    READ1(code_size);
    std::vector<size_t> sizes(nlist);
    read_ArrayInvertedLists_sizes(f, sizes);
    auto bc = std::make_shared<BlockCacheInvertedListsIOHook>(index_cache);
    return bc->read_ArrayInvertedLists(
           f, io_flags, nlist, code_size, sizes);
  } else {
    return InvertedListsIOHook::lookup(h)->read(f, io_flags);
  }
}

static void read_InvertedLists(IndexIVF* ivf, IOReader* f, int io_flags, bool use_block_cache,
                               tenann::IndexCache* index_cache) {
  InvertedLists* ils = nullptr;
  if (use_block_cache) {
    ils = read_InvertedLists_with_block_cache(f, io_flags, index_cache);
  } else {
    ils = read_InvertedLists(f, io_flags);
  }
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

static void read_ivfpq(IndexIVFPQ* ivpq, IOReader* f, uint32_t h, int io_flags, bool use_block_cache,
                       tenann::IndexCache* index_cache) {
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
    read_InvertedLists(ivpq, f, io_flags, use_block_cache, index_cache);
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

#define INVALID_OFFSET (size_t)(-1)

BlockCacheInvertedLists::BlockCacheInvertedLists(
        size_t nlist,
        size_t code_size,
        const char* filename,
        tenann::IndexCache* index_cache)
        : InvertedLists(nlist, code_size),
          filename(filename),
          totsize(0),
          index_cache(index_cache) {
  lists.resize(nlist);

  // slots starts empty
}

BlockCacheInvertedLists::BlockCacheInvertedLists(tenann::IndexCache* index_cache) :
                                                 BlockCacheInvertedLists(0, 0, "", index_cache) {}

BlockCacheInvertedLists::~BlockCacheInvertedLists() {
  // close file
  if (fd != -1) {
    close(fd);
  }
}

size_t BlockCacheInvertedLists::list_size(size_t list_no) const {
  return lists[list_no].size;
}

const uint8_t* BlockCacheInvertedLists::get_ptr(size_t list_no) const {
  tenann::IndexCacheHandle cache_handle;
  auto found = index_cache->Lookup(cache_keys[list_no], &cache_handle);
  if (found) {
    VLOG(VERBOSE_DEBUG) << "   hit cache, cache_key: " << cache_keys[list_no].c_str()
                        << ", hit_rate: " << index_cache->hit_count() * 1.0 / index_cache->lookup_count();
    auto start_ptr = static_cast<uint8_t*>(cache_handle.index_ref()->index_raw());
    return start_ptr + offset_difference[list_no];
  }

  // read file and insert

  // Calculate the offset and size for the specific list
  size_t offset = lists[list_no].offset;
  size_t size = lists[list_no].size * one_entry_size;

  // Align the offset to the block size
  size_t aligned_offset = (offset / block_size) * block_size;

  // Adjust the size to read to include the data from the aligned offset
  size_t aligned_size = ((offset_difference[list_no] + size + block_size - 1) / block_size) * block_size;

  // Allocate aligned memory
  void* buffer;
  int err = posix_memalign(&buffer, block_size, aligned_size);
  FAISS_THROW_IF_NOT_FMT(err == 0, "posix_memalign error: %d", err);

  // Seek to the aligned offset
  off_t seek_result = lseek(fd, aligned_offset, SEEK_SET);
  FAISS_THROW_IF_NOT_FMT(seek_result != -1, "lseek to aligned_offset %zu failed: %s", aligned_offset, strerror(errno));

  size_t remaining_size = totsize - aligned_offset;
  size_t expected_read_size = std::min(remaining_size, aligned_size);
  // Perform the read operation
  ssize_t read_bytes = read(fd, buffer, aligned_size);
  FAISS_THROW_IF_NOT_FMT(read_bytes == expected_read_size, "read_bytes: %zd, expected_read_size: %zu",
                         read_bytes, expected_read_size);

  auto index_ref = std::make_shared<tenann::Index>(buffer,
                                                   tenann::IndexType::kFaissIvfPqOneInvertedList,
                                                   [](void* index) { free(index); });

  index_cache->Insert(cache_keys[list_no], index_ref, &cache_handle, [read_bytes]() { return read_bytes; });
  VLOG(VERBOSE_DEBUG) << "insert cache, cache_key: " << cache_keys[list_no].c_str()
                      << ", usage: " << index_cache->memory_usage();
  return static_cast<uint8_t*>(buffer) + offset_difference[list_no];
}

const uint8_t* BlockCacheInvertedLists::get_codes(size_t list_no) const {
  if (lists[list_no].offset == INVALID_OFFSET) {
    return nullptr;
  }
  const uint8_t* ret = get_ptr(list_no);

  return ret;
}

const Index::idx_t* BlockCacheInvertedLists::get_ids(size_t list_no) const {
  if (lists[list_no].offset == INVALID_OFFSET) {
    return nullptr;
  }

  const idx_t* ret = (const idx_t*)(get_ptr(list_no) + code_size * lists[list_no].capacity);
  return ret;
}

BlockCacheInvertedListsIOHook::BlockCacheInvertedListsIOHook(tenann::IndexCache* index_cache)
        : InvertedListsIOHook("ilbc", typeid(BlockCacheInvertedLists).name()), index_cache(index_cache) {}

InvertedLists* BlockCacheInvertedListsIOHook::read_ArrayInvertedLists(
      IOReader* f,
      int /* io_flags */,
      size_t nlist,
      size_t code_size,
      const std::vector<size_t>& sizes) const {
  auto ails = new BlockCacheInvertedLists(index_cache);
  ails->filename = f->name;
  ails->nlist = nlist;
  ails->code_size = code_size;
  ails->read_only = true;
  ails->lists.resize(nlist);
  ails->cache_keys.resize(nlist);
  ails->offset_difference.resize(nlist);

  FileIOReader* reader = dynamic_cast<FileIOReader*>(f);
  FAISS_THROW_IF_NOT_MSG(reader, "only supported for File objects");
  FILE* fdesc = reader->f;
  ails->start_offset = ftell(fdesc);
  size_t o = ails->start_offset;

  ails->fd = open(f->name.c_str(), O_RDONLY | O_DIRECT);
  FAISS_THROW_IF_NOT_FMT(ails->fd != -1, "could not open file %s with O_DIRECT: %s", reader->name, strerror(errno));

  struct stat buf;
  int ret = fstat(fileno(fdesc), &buf);
  FAISS_THROW_IF_NOT_FMT(ret == 0, "fstat failed: %s", strerror(errno));
  ails->totsize = buf.st_size;
  FAISS_THROW_IF_NOT(o <= ails->totsize);

  // generate cache_keys
  // cache_key = hash(filename) + fileModificationTime + blockId
  std::string prefix = std::to_string(std::hash<std::string>{}(ails->filename)) + "_" + std::to_string(buf.st_mtime) + "_";
  for (size_t i = 0; i < nlist; i++) {
    ails->cache_keys[i] = prefix + std::to_string(i);
  }

  ails->one_entry_size = sizeof(BlockCacheInvertedLists::idx_t) + ails->code_size;
  for (size_t i = 0; i < ails->nlist; i++) {
    BlockCacheInvertedLists::List& l = ails->lists[i];
    l.size = l.capacity = sizes[i];
    l.offset = o;
    o += l.size * ails->one_entry_size;

    size_t aligned_offset = (l.offset / ails->block_size) * ails->block_size;
    ails->offset_difference[i] = l.offset - aligned_offset;
  }
  // resume normal reading of file
  fseek(fdesc, o, SEEK_SET);

  return ails;
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
    T_LOG_IF(WARNING, h != fourcc("IwPQ") && h != fourcc("IxPT"))
        << "tenann could not read ivfpq from file " << path << ": "
        << "expect magic number `IwPQ` and `IxPT` but got." << fourcc_inv_printable(h);
    if (h == fourcc("IwPQ")) {
      auto index_ivfpq = std::make_unique<IndexIvfPq>();
      // read faiss IndexIVFPQ
      VLOG(VERBOSE_DEBUG) << "use_block_cache: " << index_reader_options_.use_block_cache;
      faiss::read_ivfpq(index_ivfpq.get(), f, h, IO_FLAG, index_reader_options_.use_block_cache, index_cache());
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
    } else if (h == fourcc("IxPT")) {
      auto index_pt = std::make_unique<faiss::IndexPreTransform>();
      index_pt->own_fields = true;
      faiss::read_index_header(index_pt.get(), f);
      int nt;
      READ1(nt);
      for (int i = 0; i < nt; i++) {
        index_pt->chain.push_back(read_VectorTransform(f));
      }
      auto index_ivfpq = std::make_unique<IndexIvfPq>();
      READ1(h);
      VLOG(VERBOSE_DEBUG) << "use_block_cache: " << index_reader_options_.use_block_cache;
      faiss::read_ivfpq(index_ivfpq.get(), f, h, IO_FLAG, index_reader_options_.use_block_cache, index_cache());
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
      index_pt->index = index_ivfpq.release();
      return std::make_shared<Index>(
          index_pt.release(),      //
          IndexType::kFaissIvfPq,  //
          [](void* index) { delete static_cast<faiss::IndexPreTransform*>(index); });
    } else {
      T_LOG(INFO) << "Unknow index to tenann::reader. using faiss::reader";
      return std::make_shared<Index>(faiss::read_index(f, IO_FLAG),  //
                                     IndexType::kFaissIvfPq,         //
                                     [](void* index) { delete static_cast<faiss::Index*>(index); });
    }
  } catch (faiss::FaissException& e) {
    T_LOG(ERROR) << e.what();
  }
}

}  // namespace tenann