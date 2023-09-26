/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#pragma once

// Base
#define FAISS_SEARCHER_PARAMS_ID_SELECTOR "IDSelector"  // IDSelector*

// HNSW : Base
#define FAISS_SEARCHER_PARAMS_HNSW_EF_SEARCH "efSearch"                               // int
#define FAISS_SEARCHER_PARAMS_HNSW_CHECK_RELATIVE_DISTANCE "check_relative_distance"  // bool

// PQ : Base
#define FAISS_SEARCHER_PARAMS_SEARCH_TYPE "search_type"  // IndexPQ::Search_type_t

// IVF : Base
#define FAISS_SEARCHER_PARAMS_IVF_NPROBE "nprobe"                      // size_t
#define FAISS_SEARCHER_PARAMS_IVF_MAX_CODES "max_codes"                // size_t
#define FAISS_SEARCHER_PARAMS_IVF_QUANTIZER_PARAMS "quantizer_params"  // SearchParameters*

// IVF_PQ : IVF : Base
#define FAISS_SEARCHER_PARAMS_IVF_PQ_SCAN_TABLE_THRESHOLD "scan_table_threshold"  // size_t
#define FAISS_SEARCHER_PARAMS_IVF_PQ_POLYSEMOUS_HT "polysemous_ht"                // int