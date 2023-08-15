cmake_minimum_required(VERSION 3.5.1)
project(tenann)

set(CMAKE_CXX_STANDARD 17)

option(BUILD_TESTS "ON for build unit test or OFF for not" ON)
option(BUILD_EXAMPLES "ON for build examples or OFF for not" ON)

include_directories(
        ${CMAKE_SOURCE_DIR}/
)

set(TENANN_SOURCE tenann/version.cc)

ADD_LIBRARY(tenann ${TENANN_SOURCE})

# 设置包含路径和链接库
include_directories(${CMAKE_SOURCE_DIR}/third_party/faiss/include)
link_directories(${CMAKE_SOURCE_DIR}/third_party/faiss/lib)
# 添加 faiss 库
add_library(faiss STATIC IMPORTED)
set_target_properties(faiss PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/third_party/faiss/lib/libfaiss.a)
# 链接 faiss 库
target_link_libraries(tenann faiss)

# 链接 blas 库
find_package(BLAS REQUIRED)
target_link_libraries(tenann ${BLAS_LIBRARIES})

# 链接 OpenMP 库
find_package(OpenMP REQUIRED)
target_link_libraries(tenann OpenMP::OpenMP_CXX)

if (BUILD_EXAMPLES STREQUAL "ON")
    add_subdirectory(examples)
endif ()

if (BUILD_TESTS STREQUAL "ON")
    add_subdirectory(test)
endif ()