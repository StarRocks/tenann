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

#pragma once

#include "tenann/common/json.h"
#include "tenann/index/index_reader.h"
#include "tenann/index/index_cache.h"

#include <faiss/Index.h>
#include <faiss/IndexIVF.h>
#include <faiss/index_io.h>
#include <faiss/invlists/InvertedListsIOHook.h>
#include <faiss/invlists/OnDiskInvertedLists.h>

namespace faiss {

Index* read_index_with_block_cache(const char* fname);

struct BlockCacheInvertedLists : InvertedLists {
    using List = OnDiskOneList;

    // size nlist
    std::vector<List> lists;
    std::vector<std::string> cache_keys;
    std::vector<size_t> offset_difference;

    std::string filename;
    size_t one_entry_size;
    size_t totsize;
    size_t start_offset;  // inserted lists start offset
    size_t block_size = 4096;  // block size
    bool read_only; /// are inverted lists mapped read-only
    int fd = -1;
    tenann::IndexCache* index_cache = nullptr;

    BlockCacheInvertedLists(size_t nlist, size_t code_size, const char* filename, tenann::IndexCache* index_cache);

    size_t list_size(size_t list_no) const override;
    const uint8_t* get_ptr(size_t list_no) const;
    const uint8_t* get_codes(size_t list_no) const override;
    const idx_t* get_ids(size_t list_no) const override;

    size_t add_entries(
        size_t list_no,
        size_t n_entry,
        const idx_t* ids,
        const uint8_t* code) { return 0; }

    void update_entries(
        size_t list_no,
        size_t offset,
        size_t n_entry,
        const idx_t* ids,
        const uint8_t* code) {}

    void resize(size_t list_no, size_t new_size) {}

    ~BlockCacheInvertedLists() override;

    // private

    // empty constructor for the I/O functions
    BlockCacheInvertedLists(tenann::IndexCache* index_cache);
};

struct BlockCacheInvertedListsIOHook : InvertedListsIOHook {
    BlockCacheInvertedListsIOHook(tenann::IndexCache* index_cache);
    void write(const InvertedLists* ils, IOWriter* f) const {}
    InvertedLists* read(IOReader* f, int io_flags) const { return nullptr; }
    InvertedLists* read_ArrayInvertedLists(
            IOReader* f,
            int io_flags,
            size_t nlist,
            size_t code_size,
            const std::vector<size_t>& sizes) const override;

    tenann::IndexCache* index_cache = nullptr;
};

}  // namespace faiss

namespace tenann {

class IndexIvfPqReader : public IndexReader {
 public:
  using IndexReader::IndexReader;
  virtual ~IndexIvfPqReader();

  T_FORBID_COPY_AND_ASSIGN(IndexIvfPqReader);
  T_FORBID_MOVE(IndexIvfPqReader);

  IndexRef ReadIndexFile(const std::string& path) override;
};

}  // namespace tenann