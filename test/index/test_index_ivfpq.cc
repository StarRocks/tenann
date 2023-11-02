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

#include "faiss/IndexFlat.h"
#include "faiss/utils/distances.h"
#include "gtest/gtest.h"
#include "tenann/index/internal/index_ivfpq.h"
#include "tenann/util/random.h"

const float float_diff_threshold = 0.000001;

TEST(InternalIvfPq, test_reconstruction_error) {
  const int dim = 8;
  const int m = 2;
  const int nlist = 2;
  const int nbits = 8;
  const int nb = 1024;

  auto base = tenann::RandomVectors(nb, dim, 0);
  faiss::IndexFlatL2 coarse_quantizer(dim);
  tenann::IndexIvfPq ivfpq(&coarse_quantizer, dim, nlist, m, nbits);
  ivfpq.train(nb, base.data());
  ivfpq.add(nb, base.data());

  std::vector<float> recons(dim);
  for (int list_no = 0; list_no < nlist; list_no++) {
    auto size = ivfpq.get_list_size(list_no);
    for (auto offset = 0; offset < size; offset++) {
      auto id = ivfpq.invlists->get_single_id(list_no, offset);
      ivfpq.reconstruct_from_offset(list_no, offset, recons.data());
      float actual_squared_error = faiss::fvec_L2sqr(base.data() + dim * id, recons.data(), dim);
      float actual_error = sqrtf(actual_squared_error);
      EXPECT_LE(actual_error - ivfpq.reconstruction_errors[list_no][offset], float_diff_threshold);
    }
  }
}