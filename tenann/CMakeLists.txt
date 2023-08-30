# Copyright (c) Tencent, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set(TENANN_SRC
    version.cc
    builder/index_builder.cc
    builder/faiss_hnsw_index_builder.cc
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
    store/index_meta.cc
    store/lru_cache.cc
)

set(TENANN_HEADERS
    version.h
    builder/index_builder.h
    builder/faiss_hnsw_index_builder.h
    common/error.h
    common/json.hpp
    common/logging.h
    common/macros.h
    common/primitive_type.h
    common/seq_view.h
    common/type_traits.h
    common/typed_seq_view.h
    factory/ann_searcher_factory.h
    factory/index_factory_trait.h
    factory/index_factory.h
    index/index_cache.h
    index/index_reader.h
    index/faiss_index_reader.h
    index/index_writer.h
    index/faiss_index_writer.h
    index/index.h
    searcher/ann_searcher.h
    searcher/faiss_hnsw_ann_searcher.h
    searcher/searcher.h
    store/index_meta.h
    store/index_type.h
    store/lru_cache.h
)

set(CMAKE_CXX_FLAGS "-static-libstdc++ -static-libgcc ${CMAKE_CXX_FLAGS} -gdwarf-4")

# 生成 tenann 库
ADD_LIBRARY(tenann STATIC ${TENANN_SRC})

target_include_directories(tenann
  INTERFACE
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

# 添加 faiss 库
find_package(faiss REQUIRED PATHS "${PROJECT_SOURCE_DIR}/thirdparty/installed/lib/cmake/faiss")

# 添加 faiss_avx2 库
find_package(faiss_avx2 REQUIRED PATHS "${PROJECT_SOURCE_DIR}/thirdparty/installed/lib/cmake/faiss")

# 添加 lapack 库
find_package(LAPACK REQUIRED PATHS "${PROJECT_SOURCE_DIR}/thirdparty/installed/lib/cmake/lapack")

# 添加 OpenMP 库
find_package(OpenMP REQUIRED)

# 添加 gfortran 动态库依赖
# TODO: 目前指定动态库软链名，后续考虑修改
find_file(GFORTRAN_LIBRARY NAMES libgfortran.so.3 PATHS /usr/lib /usr/lib64/)

# 链接依赖库
set(TENANN_LINK_LIBS
    -static-libstdc++
    -static-libgcc
    faiss
    faiss_avx2
    ${LAPACK_LIBRARIES}
    OpenMP::OpenMP_CXX
    ${GFORTRAN_LIBRARY}
)

target_link_libraries(tenann
    ${TENANN_LINK_LIBS}
)

install(TARGETS tenann
    EXPORT tenann-targets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/tenann/include)
foreach(header ${TENANN_HEADERS})
  get_filename_component(dir ${header} DIRECTORY )
  file(COPY ${header} DESTINATION ${CMAKE_BINARY_DIR}/tenann/include/${dir})
  install(FILES ${header}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tenann/${dir}
  )
endforeach()

# 使用 CMake 的 CMakePackageConfigHelpers 模块来生成一个基本的包版本文件
include(CMakePackageConfigHelpers)
WRITE_BASIC_CONFIG_VERSION_FILE(
  "${PROJECT_BINARY_DIR}/cmake/tenann-config-version.cmake"
  VERSION ${CMAKE_PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion
)

configure_file(${PROJECT_SOURCE_DIR}/cmake/tenann-config.cmake.in
  ${PROJECT_BINARY_DIR}/cmake/tenann-config.cmake
  COPYONLY
)

install(FILES ${PROJECT_BINARY_DIR}/cmake/tenann-config.cmake
  ${PROJECT_BINARY_DIR}/cmake/tenann-config-version.cmake
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/tenann
)

install(EXPORT tenann-targets
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/tenann
)