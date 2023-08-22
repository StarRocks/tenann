cmake_minimum_required(VERSION 3.5.1)
project(tenann)

set(CMAKE_CXX_STANDARD 17)

option(BUILD_TESTS "ON for build unit test or OFF for not" ON)
option(BUILD_EXAMPLES "ON for build examples or OFF for not" ON)

include_directories(
        ${CMAKE_CURRENT_LIST_DIR}/
        ${CMAKE_CURRENT_LIST_DIR}/third_party/
)

set(TENANN_SOURCE
        tenann/builder/index_builder.cc
        tenann/factory/index_factory.cc
        tenann/factory/ann_searcher_factory.cc
        tenann/index/index.cc
        tenann/index/index_reader.cc
        tenann/index/index_writer.cc
        tenann/searcher/searcher.cc
        tenann/searcher/faiss_hnsw_ann_searcher.cc
        tenann/store/index_cache.cc
        tenann/store/index_meta.cc
        tenann/store/lru_cache.cc)

add_library(tenann STATIC ${TENANN_SOURCE})

#file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/build64_release/lib)
#file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/build64_release/include)
#file(GLOB HEADER_FILES ${CMAKE_SOURCE_DIR}/tenann/*.h)
#file(COPY ${HEADER_FILES} DESTINATION ${CMAKE_SOURCE_DIR}/build64_release/include)
#
#ADD_LIBRARY(tenann SHARED ${TENANN_SOURCE})
#set_target_properties(tenann PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build64_release/lib)
#
## 设置包含路径和链接库
#include_directories(${CMAKE_SOURCE_DIR}/third_party/faiss/include)
#link_directories(${CMAKE_SOURCE_DIR}/third_party/faiss/lib)
## 添加 faiss 库
#add_library(faiss SHARED IMPORTED)
#set_target_properties(faiss PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/third_party/faiss/lib/libfaiss.a)
## 链接 faiss 库
#target_link_libraries(tenann faiss)
#
## 链接 blas 库
#find_package(BLAS REQUIRED COMPONENTS BLAS)
#target_link_libraries(tenann ${BLAS_LIBRARIES})
#
## 链接 OpenMP 库
#find_package(OpenMP REQUIRED)
#target_link_libraries(tenann OpenMP::OpenMP_CXX)

if (BUILD_EXAMPLES STREQUAL "ON")
    add_subdirectory(examples)
endif ()

if (BUILD_TESTS STREQUAL "ON")
    add_subdirectory(test)
endif ()
