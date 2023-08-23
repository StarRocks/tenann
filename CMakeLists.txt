cmake_minimum_required(VERSION 3.5.1)

set(TENANN_LANGUAGES CXX)

include_directories(
        ${CMAKE_CURRENT_LIST_DIR}/
)

project(tenann
    VERSION 0.0.1
    DESCRIPTION "A library for fast and flexible approximate nearest neighbor search."
    HOMEPAGE_URL "https://git.woa.com/petrizhang/tenann"
    LANGUAGES ${TENANN_LANGUAGES})
include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

# 编译选项
option(BUILD_TESTS "ON for build unit test or OFF for not" ON)
option(BUILD_EXAMPLES "ON for build examples or OFF for not" ON)

add_subdirectory(tenann)

if (BUILD_EXAMPLES STREQUAL "ON")
    add_subdirectory(examples)
endif ()

# CTest must be included in the top level to enable `make test` target.
include(CTest)
if (BUILD_TESTS STREQUAL "ON")
    add_subdirectory(test)
endif ()
