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

cmake_minimum_required(VERSION 3.5.1)

set(TENANN_LANGUAGES CXX)

project(tenann
    VERSION 0.0.2
    DESCRIPTION "A library for fast and flexible approximate nearest neighbor search."
    HOMEPAGE_URL "https://git.woa.com/petrizhang/tenann"
    LANGUAGES ${TENANN_LANGUAGES})
# include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)

if(DEFINED ENV{TENANN_THIRDPARTY})
    set(TENANN_THIRDPARTY $ENV{TENANN_THIRDPARTY})
else()
    set(TENANN_THIRDPARTY "${PROJECT_SOURCE_DIR}/thirdparty")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libstdc++ -static-libgcc -gdwarf-4")

# Build options
option(WITH_TESTS "ON for build unit test or OFF for not" ON)
option(WITH_EXAMPLES "ON for build examples or OFF for not" ON)
option(SUPPORT_AVX2 "ON for build tenann with avx2 support" OFF)

if (WITH_TESTS STREQUAL "ON")
    # Open code coverage before add_subdirectory(tenann)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGTEST_USE_OWN_TR1_TUPLE=0 -fprofile-arcs -ftest-coverage")
endif ()

include_directories(
    ${CMAKE_CURRENT_LIST_DIR}/
    ${TENANN_THIRDPARTY}/installed/include)
add_subdirectory(tenann)

if (WITH_EXAMPLES STREQUAL "ON")
    add_subdirectory(examples)
endif ()

if (WITH_TESTS STREQUAL "ON")
    include(CTest)
    add_subdirectory(test)
endif ()

########################################################
# TenANN install
########################################################

install(TARGETS tenann
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)

if(SUPPORT_AVX2 STREQUAL "ON")
    install(TARGETS tenann_avx2
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)
endif()

install(DIRECTORY tenann
        DESTINATION include
        FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.hpp"
        PATTERN "examples" EXCLUDE)