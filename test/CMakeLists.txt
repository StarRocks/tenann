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

set(TENANN_TEST_SRC
    tenann_test.cc
    faiss_test_base.cc
    builder/test_faiss_index_builder.cc
    builder/test_faiss_hnsw_index_builder.cc
    builder/test_faiss_hnsw_quantized_builder.cc
    builder/test_faiss_ivf_pq_index_builder.cc
    builder/test_faiss_ivf_sq_index_builder.cc
    index/test_index.cc
    index/test_index_ivfpq.cc
    index/test_hnsw_quantizer_params.cc
    index/test_hnsw_factory_string.cc
    index/test_ivf_sq_params.cc
    searcher/test_faiss_hnsw_ann_searcher.cc
    searcher/test_faiss_ivf_pq_ann_searcher.cc
    searcher/test_faiss_ivf_sq_ann_searcher.cc
    searcher/test_ivf_pq_range_search.cc
    searcher/test_range_search.cc
    store/test_index_meta.cc
    store/test_lru_cache.cc
    store/test_index_file_reader.cc
    store/test_index_file_writer.cc
    searcher/test_file_reader_search.cc
    thirdparty/test_fmt.cc
    util/test_runtime_profile.cc
    util/test_bruteforce_search.cc
    util/test_distance_util.cc
)

add_executable(tenann_test ${TENANN_TEST_SRC})

# add gtest lib
set(GTest_DIR "${TENANN_THIRDPARTY}/installed/lib/cmake/GTest" CACHE PATH "gtest search path")
find_package(GTest REQUIRED)

target_link_libraries(tenann_test PRIVATE tenann GTest::gtest)
include(GoogleTest)

# add tenann_test to ctest, log path: build_Release/Testing/Temporary/LastTest.log
gtest_discover_tests(tenann_test)

add_custom_target(coverage
  COMMAND lcov --capture --directory . --output-file coverage.info
  COMMAND lcov --extract coverage.info '*/tenann/tenann/*' --output-file coverage.info
  COMMAND lcov --remove coverage.info '*/tenann/tenann/common*' --output-file coverage.info
  COMMAND lcov --remove coverage.info '*/tenann/tenann/util*' --output-file coverage.info
  COMMAND lcov --remove coverage.info '*/tenann/tenann/index/internal*' --output-file coverage.info
  COMMAND lcov --list coverage.info
  COMMAND genhtml coverage.info --output-directory coverage_html --rc genhtml_hi_limit=50 --rc genhtml_med_limit=30
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  COMMENT "Generating code coverage report")
