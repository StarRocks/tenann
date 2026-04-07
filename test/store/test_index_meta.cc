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
#include <fstream>

#include "tenann/index/parameters.h"
#include "test/faiss_test_base.h"

namespace tenann {
TEST(IndexMetaTest, BasicTest) {
  // Create an empty IndexMeta object
  IndexMeta index_meta;

  // Set meta_version
  EXPECT_THROW(index_meta.meta_version(), Error);
  index_meta.SetMetaVersion(1);
  EXPECT_EQ(index_meta.meta_version(), 1);

  // Set index_family
  EXPECT_THROW(index_meta.index_family(), Error);
  index_meta.SetIndexFamily(IndexFamily::kVectorIndex);
  EXPECT_EQ(index_meta.index_family(), static_cast<int>(IndexFamily::kVectorIndex));

  // Set index_type
  EXPECT_THROW(index_meta.index_type(), Error);
  index_meta.SetIndexType(IndexType::kFaissHnsw);
  EXPECT_EQ(index_meta.index_type(), static_cast<int>(IndexType::kFaissHnsw));

  // Set common_params
  index_meta.common_params()["dim"] = 128;
  EXPECT_EQ(index_meta.common_params()["dim"], 128);

  // Set index_params
  index_meta.index_params()["ntrees"] = 10;
  EXPECT_EQ(index_meta.index_params()["ntrees"], 10);

  // Set search_params
  index_meta.search_params()["nprobe"] = 32;
  EXPECT_EQ(index_meta.search_params()["nprobe"], 32);

  // Set extra_params
  index_meta.extra_params()["key"] = "value";
  EXPECT_EQ(index_meta.extra_params()["key"], "value");
}

TEST(IndexMetaTests, CheckIntegrity) {
  std::string error_msg;
  tenann::IndexMeta index_meta;
  EXPECT_FALSE(index_meta.CheckIntegrity(&error_msg));
  index_meta.SetMetaVersion(1);

  EXPECT_FALSE(index_meta.CheckIntegrity(&error_msg));
  index_meta.SetIndexFamily(tenann::IndexFamily::kVectorIndex);

  EXPECT_FALSE(index_meta.CheckIntegrity(&error_msg));
  index_meta.SetIndexType(tenann::IndexType::kFaissHnsw);

  EXPECT_TRUE(index_meta.CheckIntegrity(&error_msg));
}

TEST(IndexMetaTest, SerializeAndDeserialize) {
  // Create an IndexMeta object
  IndexMeta index_meta;
  // Set meta_version
  index_meta.SetMetaVersion(1);
  // Set index_family
  index_meta.SetIndexFamily(IndexFamily::kVectorIndex);
  // Set index_type
  index_meta.SetIndexType(IndexType::kFaissHnsw);
  // Set common_params
  index_meta.common_params()["dim"] = 128;
  // Set index_params
  index_meta.index_params()["ntrees"] = 10;
  // Set search_params
  index_meta.search_params()["nprobe"] = 32;
  // Set extra_params
  index_meta.extra_params()["key"] = "value";

  // Serialize IndexMeta object
  std::vector<uint8_t> buffer = index_meta.Serialize();

  // Deserialize IndexMeta object
  IndexMeta deserialized_index_meta = IndexMeta::Deserialize(buffer);

  // Verify the deserialized IndexMeta object matches the original
  EXPECT_EQ(index_meta.meta_json(), deserialized_index_meta.meta_json());
  EXPECT_EQ(deserialized_index_meta.meta_version(), index_meta.meta_version());
  EXPECT_EQ(deserialized_index_meta.index_family(), index_meta.index_family());
  EXPECT_EQ(deserialized_index_meta.index_type(), index_meta.index_type());
  EXPECT_EQ(deserialized_index_meta.common_params(), index_meta.common_params());
  EXPECT_EQ(deserialized_index_meta.index_params(), index_meta.index_params());
  EXPECT_EQ(deserialized_index_meta.search_params(), index_meta.search_params());
  EXPECT_EQ(deserialized_index_meta.extra_params(), index_meta.extra_params());
}

TEST(IndexMetaTest, StrigifyAndParse) {
  // Create an IndexMeta object
  IndexMeta index_meta;
  // Set meta_version
  index_meta.SetMetaVersion(1);
  // Set index_family
  index_meta.SetIndexFamily(IndexFamily::kVectorIndex);
  // Set index_type
  index_meta.SetIndexType(IndexType::kFaissHnsw);
  // Set common_params
  index_meta.common_params()["dim"] = 128;
  // Set index_params
  index_meta.index_params()["ntrees"] = 10;
  // Set search_params
  index_meta.search_params()["nprobe"] = 32;
  // Set extra_params
  index_meta.extra_params()["key"] = "value";

  // Serialize IndexMeta object
  std::string buffer = index_meta.Stringify();

  // Deserialize IndexMeta object
  IndexMeta deserialized_index_meta = IndexMeta::Parse(buffer);

  // Verify the deserialized IndexMeta object matches the original
  EXPECT_EQ(index_meta.meta_json(), deserialized_index_meta.meta_json());
  EXPECT_EQ(deserialized_index_meta.meta_version(), index_meta.meta_version());
  EXPECT_EQ(deserialized_index_meta.index_family(), index_meta.index_family());
  EXPECT_EQ(deserialized_index_meta.index_type(), index_meta.index_type());
  EXPECT_EQ(deserialized_index_meta.common_params(), index_meta.common_params());
  EXPECT_EQ(deserialized_index_meta.index_params(), index_meta.index_params());
  EXPECT_EQ(deserialized_index_meta.search_params(), index_meta.search_params());
  EXPECT_EQ(deserialized_index_meta.extra_params(), index_meta.extra_params());
}

TEST(IndexMetaTest, WriteAndRead) {
  // Create an IndexMeta object
  tenann::IndexMeta index_meta;
  index_meta.SetMetaVersion(1);
  index_meta.SetIndexFamily(tenann::IndexFamily::kTextIndex);
  index_meta.SetIndexType(tenann::IndexType::kFaissIvfPq);
  index_meta.common_params()["dim"] = 128;
  index_meta.index_params()["nprobe"] = 32;
  index_meta.search_params()["nprobe"] = 32;
  index_meta.extra_params()["metric_type"] = 1;

  // Write IndexMeta object to file
  std::string file_path = "/tmp/test_index_meta.json";
  std::remove(file_path.c_str());
  EXPECT_TRUE(index_meta.Write(file_path));

  // Read IndexMeta object from file
  tenann::IndexMeta read_index_meta = IndexMeta::Read(file_path);

  // Verify the read IndexMeta object matches the original
  EXPECT_EQ(index_meta.meta_json(), read_index_meta.meta_json());
  EXPECT_EQ(read_index_meta.meta_version(), 1);
  EXPECT_EQ(read_index_meta.index_family(), tenann::IndexFamily::kTextIndex);
  EXPECT_EQ(read_index_meta.index_type(), tenann::IndexType::kFaissIvfPq);
  EXPECT_EQ(read_index_meta.common_params()["dim"], 128);
  EXPECT_EQ(read_index_meta.index_params()["nprobe"], 32);
  EXPECT_EQ(read_index_meta.search_params()["nprobe"], 32);
  EXPECT_EQ(read_index_meta.extra_params()["metric_type"], 1);
}
}  // namespace tenann