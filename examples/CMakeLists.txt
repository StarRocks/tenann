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

# hello world example
add_executable(helloworld_example helloworld_example.cc)
target_link_libraries(helloworld_example tenann)

# meta example
add_executable(meta_example meta_example.cc)
target_link_libraries(meta_example tenann)

# index factory example
add_executable(index_factory_example index_factory_example.cc)
target_link_libraries(index_factory_example tenann)

# faiss hnsw example
add_executable(faiss_hnsw_example faiss_hnsw_example.cc)
target_link_libraries(faiss_hnsw_example tenann)

# error handling example
add_executable(error_handling_example error_handling_example.cc)
target_link_libraries(error_handling_example tenann)

# index cache example
add_executable(index_cache_example index_cache_example.cc)
target_link_libraries(index_cache_example tenann)