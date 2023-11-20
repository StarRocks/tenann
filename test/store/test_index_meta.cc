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
  // 创建一个空的 IndexMeta 对象
  IndexMeta index_meta;

  // 设置 meta_version
  EXPECT_THROW(index_meta.meta_version(), Error);
  index_meta.SetMetaVersion(1);
  EXPECT_EQ(index_meta.meta_version(), 1);

  // 设置 index_family
  EXPECT_THROW(index_meta.index_family(), Error);
  index_meta.SetIndexFamily(IndexFamily::kVectorIndex);
  EXPECT_EQ(index_meta.index_family(), static_cast<int>(IndexFamily::kVectorIndex));

  // 设置 index_type
  EXPECT_THROW(index_meta.index_type(), Error);
  index_meta.SetIndexType(IndexType::kFaissHnsw);
  EXPECT_EQ(index_meta.index_type(), static_cast<int>(IndexType::kFaissHnsw));

  // 设置 common_params
  index_meta.common_params()["dim"] = 128;
  EXPECT_EQ(index_meta.common_params()["dim"], 128);

  // 设置 index_params
  index_meta.index_params()["ntrees"] = 10;
  EXPECT_EQ(index_meta.index_params()["ntrees"], 10);

  // 设置 search_params
  index_meta.search_params()["nprobe"] = 32;
  EXPECT_EQ(index_meta.search_params()["nprobe"], 32);

  // 设置 extra_params
  index_meta.extra_params()["key"] = "value";
  EXPECT_EQ(index_meta.extra_params()["key"], "value");
}

TEST(IndexMetaTests, CheckIntegrity) {
  std::string error_msg;
  tenann::IndexMeta index_meta;
  EXPECT_THROW(index_meta.CheckIntegrity(&error_msg), Error);
  index_meta.SetMetaVersion(1);

  EXPECT_THROW(index_meta.CheckIntegrity(&error_msg), Error);
  index_meta.SetIndexFamily(tenann::IndexFamily::kVectorIndex);

  EXPECT_THROW(index_meta.CheckIntegrity(&error_msg), Error);
  index_meta.SetIndexType(tenann::IndexType::kFaissHnsw);

  EXPECT_TRUE(index_meta.CheckIntegrity(&error_msg));
  EXPECT_TRUE(error_msg.empty());
}

TEST(IndexMetaTest, SerializeAndDeserialize) {
  // 创建一个 IndexMeta 对象
  IndexMeta index_meta;
  // 设置 meta_version
  index_meta.SetMetaVersion(1);
  // 设置 index_family
  index_meta.SetIndexFamily(IndexFamily::kVectorIndex);
  // 设置 index_type
  index_meta.SetIndexType(IndexType::kFaissHnsw);
  // 设置 common_params
  index_meta.common_params()["dim"] = 128;
  // 设置 index_params
  index_meta.index_params()["ntrees"] = 10;
  // 设置 search_params
  index_meta.search_params()["nprobe"] = 32;
  // 设置 extra_params
  index_meta.extra_params()["key"] = "value";

  // 序列化 IndexMeta 对象
  std::vector<uint8_t> buffer = index_meta.Serialize();

  // 反序列化 IndexMeta 对象
  IndexMeta deserialized_index_meta = IndexMeta::Deserialize(buffer);

  // 检查反序列化后的 IndexMeta 对象与原对象是否相等
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
  // 创建一个 IndexMeta 对象
  tenann::IndexMeta index_meta;
  index_meta.SetMetaVersion(1);
  index_meta.SetIndexFamily(tenann::IndexFamily::kTextIndex);
  index_meta.SetIndexType(tenann::IndexType::kFaissIvfPq);
  index_meta.common_params()["dim"] = 128;
  index_meta.index_params()["nprobe"] = 32;
  index_meta.search_params()["nprobe"] = 32;
  index_meta.extra_params()["metric_type"] = 1;

  // 将 IndexMeta 对象写入文件
  std::string file_path = "/tmp/test_index_meta.json";
  std::remove(file_path.c_str());
  EXPECT_TRUE(index_meta.Write(file_path));

  // 从文件中读取 IndexMeta 对象
  tenann::IndexMeta read_index_meta = IndexMeta::Read(file_path);

  // 检查读取的 IndexMeta 对象是否与原始对象相同
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