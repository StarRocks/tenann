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
    builder/index_builder.cc
    builder/faiss_index_builder.cc
    builder/faiss_index_builder_with_buffer.cc
    builder/faiss_hnsw_index_builder.cc
    builder/faiss_ivf_pq_index_builder.cc
    common/logging.cc
    factory/index_factory.cc
    factory/ann_searcher_factory.cc
    index/index.cc
    index/index_cache.cc
    index/index_reader.cc
    index/faiss_index_reader.cc
    index/index_writer.cc
    index/faiss_index_writer.cc
    searcher/searcher.cc
    searcher/faiss_hnsw_ann_searcher.cc
    searcher/faiss_ivf_pq_ann_searcher.cc
    store/index_meta.cc
    store/lru_cache.cc
    util/runtime_profile.cc
    util/spinlock.cc
)

# TenANN library target
add_library(tenann STATIC ${TENANN_SRC})

########################################################
# Thrid-party dependicies
########################################################

# fmt
add_library(fmt STATIC IMPORTED)
set_target_properties(fmt PROPERTIES IMPORTED_LOCATION ${TENANN_THIRDPARTY}/installed/lib64/libfmt.a)

# faiss
find_package(faiss REQUIRED PATHS "${TENANN_THIRDPARTY}/installed/lib/cmake/faiss")

# faiss_avx2
find_package(faiss_avx2 REQUIRED PATHS "${TENANN_THIRDPARTY}/installed/lib/cmake/faiss")

# lapack
find_package(LAPACK REQUIRED PATHS "${TENANN_THIRDPARTY}/installed/lib/cmake/lapack")

# openmp
find_package(OpenMP REQUIRED)

set(WL_START_GROUP "-Wl,--start-group")
set(WL_END_GROUP "-Wl,--end-group")

# TenANN link options
set(TENANN_LINK_LIBS
    ${WL_START_GROUP}
    fmt
    faiss
    # faiss_avx2 # temporaly disable avx2 version
    ${LAPACK_LIBRARIES}
    OpenMP::OpenMP_CXX
    -l:libgfortran.a
    -l:libquadmath.a
    -lm
    ${WL_END_GROUP}
)

target_link_libraries(tenann
    ${TENANN_LINK_LIBS})