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

if (BUILD_EXAMPLES STREQUAL "ON")
    add_subdirectory(examples)
endif ()

if (BUILD_TESTS STREQUAL "ON")
    add_subdirectory(test)
endif ()