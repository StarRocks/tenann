# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

set(TENANN_SRC
    version.cc
    bench/evaluator.cc
    bench/range_search_evaluator.cc
    bench/topk_search_evaluator.cc
    builder/index_builder.cc
    builder/faiss_index_builder.cc
    builder/faiss_index_builder_with_buffer.cc
    builder/faiss_hnsw_index_builder.cc
    builder/faiss_ivf_pq_index_builder.cc
    builder/faiss_ivf_sq_index_builder.cc
    common/logging.cc
    factory/index_factory.cc
    factory/ann_searcher_factory.cc
    index/internal/metric_util.cc
    index/internal/index_ivfpq.cc
    index/index_ivfpq_writer.cc
    index/index_ivfpq_reader.cc
    index/index.cc
    index/default_index_cache.cc
    index/index_cache.cc
    index/index_reader.cc
    index/faiss_index_reader.cc
    index/index_writer.cc
    index/faiss_index_writer.cc
    searcher/id_filter.cc
    searcher/searcher.cc
    searcher/ann_searcher.cc
    searcher/faiss_hnsw_ann_searcher.cc
    searcher/faiss_ivf_pq_ann_searcher.cc
    searcher/faiss_ivf_sq_ann_searcher.cc
    store/index_meta.cc
    store/lru_cache.cc
    util/runtime_profile.cc
    util/spinlock.cc
    util/threads.cc
)

# TenANN library target
add_library(tenann STATIC ${TENANN_SRC})

########################################################
# Thrid-party dependicies
########################################################

# fmt
add_library(fmt STATIC IMPORTED)
set_target_properties(fmt PROPERTIES IMPORTED_LOCATION ${TENANN_THIRDPARTY}/installed/lib64/libfmt.a)

# faiss is built with FAISS_OPT_LEVEL=dd, so this single archive carries every
# per-ISA kernel and selects one at runtime. There is no per-ISA variant to import.
add_library(faiss STATIC IMPORTED)
set_target_properties(faiss PROPERTIES IMPORTED_LOCATION ${TENANN_THIRDPARTY}/installed/lib/libfaiss.a)

# openblas
find_package(OpenBLAS REQUIRED PATHS "${TENANN_THIRDPARTY}/installed/lib/cmake/openblas")

# openmp
find_package(OpenMP REQUIRED)

set(WL_START_GROUP "-Wl,--start-group")
set(WL_END_GROUP "-Wl,--end-group")

# TenANN link options
set(TENANN_LINK_LIBS
    fmt
    faiss
    ${OpenBLAS_LIBRARIES}
    OpenMP::OpenMP_CXX
    -lm)

target_link_libraries(tenann
    ${WL_START_GROUP}
    ${TENANN_LINK_LIBS}
    ${WL_END_GROUP})
