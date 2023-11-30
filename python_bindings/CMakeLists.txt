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

set(TENANN_PYTHON_BINDINGS_SRC
    bindings.cc
)

# Include Python headers
execute_process(COMMAND /bin/bash -c "python3-config --includes | awk -F '-I' '{print $2}'"
                OUTPUT_VARIABLE python_includes
                OUTPUT_STRIP_TRAILING_WHITESPACE)
include_directories(${python_includes})

if(NOT WITH_AVX2 STREQUAL "ON")
  pybind11_add_module(tenann_py ${TENANN_PYTHON_BINDINGS_SRC})
  target_link_libraries(tenann_py PRIVATE tenann)
  set_target_properties(tenann PROPERTIES POSITION_INDEPENDENT_CODE ON)
else ()
  pybind11_add_module(tenann_py ${TENANN_PYTHON_BINDINGS_SRC})
  target_link_libraries(tenann_py PRIVATE tenann_avx2 ${TENANN_AVX2_LINK_LIBS})
  set_target_properties(tenann_avx2 PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

# Install tenann_py module
execute_process(COMMAND /bin/bash -c "python3 -m site | grep lib-dynload | awk -F \"'\" '{print $2}'"
                OUTPUT_VARIABLE python_site_path
                OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS "Python site path: ${python_site_path}")
install(TARGETS tenann_py DESTINATION ${python_site_path})