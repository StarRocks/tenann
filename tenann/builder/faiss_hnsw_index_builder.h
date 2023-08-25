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

#include "tenann/builder/index_builder.h"

namespace tenann {

class FaissHnswIndexBuilder : public IndexBuilder {
 public:
  FaissHnswIndexBuilder(const IndexMeta& meta);
  FaissHnswIndexBuilder() = delete;
  virtual ~FaissHnswIndexBuilder() = default;

  T_FORBID_COPY_AND_ASSIGN(FaissHnswIndexBuilder);
  T_FORBID_MOVE(FaissHnswIndexBuilder);

 protected:
  /// Use the given primary key column as vector id.
  void BuildWithPrimaryKeyImpl(const std::vector<SeqView>& input_columns,
                               int primary_key_column_index) override;

  /// Use the row number as vector id.
  void BuildImpl(const std::vector<SeqView>& input_columns) override;
};

}  // namespace tenann