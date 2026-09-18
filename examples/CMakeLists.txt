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

FILE(GLOB examples ./*.cc)

FOREACH (path ${examples})
    get_filename_component(name ${path} NAME_WE)
    add_executable(${name} ${path})
    target_link_libraries(${name} tenann ${TENANN_LINK_LIBS})

    add_executable("${name}_avx2" ${path})
    if(NOT WITH_AVX2 STREQUAL "ON")
        set_target_properties("${name}_avx2" PROPERTIES EXCLUDE_FROM_ALL TRUE)
    endif()
    target_compile_options("${name}_avx2" PRIVATE ${FAISS_AVX2_CXX_FLAGS})
    target_link_libraries("${name}_avx2" tenann_avx2 ${TENANN_AVX2_LINK_LIBS})
ENDFOREACH ()