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

#include <iostream>
#include <vector>

#include "tenann/common/typed_seq_view.h"

int main(int argc, char const* argv[]) {
  using namespace tenann;
  constexpr const int n = 100;
  std::vector<float> data(100);
  for (int i = 0; i < n; i++) {
    data[i] = i;
  }
  std::vector<uint32_t> offsets(11);
  offsets[0] = 0;
  for (int i = 1; i < 11; i++) {
    offsets[i] = offsets[i - 1] + 10;
  }

  TypedVlArraySeqView typed_seq_view(data.data(), offsets.data(), 10);
  int i = 0;

  for (auto slice : typed_seq_view) {
    std::cout << "Slice " << i << ":";
    for (int j = 0; j < slice.size; j++) {
      std::cout << slice.data[j] << ",";
    }
    std::cout << "\n";
    i += 1;
  }

  VlArraySeqView seq_view{.data = reinterpret_cast<uint8_t*>(data.data()),
                          .offsets = offsets.data(),
                          .size = 10,
                          .elem_type = PrimitiveType::kFloatType};

  TypedVlArraySeqView<float> typed_seq_view1(seq_view);

  i = 0;
  for (auto slice : typed_seq_view1) {
    std::cout << "Slice " << i << ":";
    for (int j = 0; j < slice.size; j++) {
      std::cout << slice.data[j] << ",";
    }
    std::cout << "\n";
    i += 1;
  }

  return 0;
}
