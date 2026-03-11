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

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "tenann/index/custom_io_reader.h"
#include "tenann/store/index_file_reader.h"

namespace tenann {

/// A local-file-based IndexFileReader for testing purposes.
/// Wraps POSIX FILE* to simulate reading from any file system.
class LocalIndexFileReader : public IndexFileReader {
 public:
  explicit LocalIndexFileReader(const std::string& path) : filename_(path), position_(0) {
    fp_ = fopen(path.c_str(), "rb");
    if (fp_) {
      fseek(fp_, 0, SEEK_END);
      file_size_ = ftell(fp_);
      fseek(fp_, 0, SEEK_SET);
    } else {
      file_size_ = 0;
    }
  }

  ~LocalIndexFileReader() override {
    if (fp_) fclose(fp_);
  }

  int64_t Read(void* data, int64_t count) override {
    if (!fp_) return -1;
    fseek(fp_, position_, SEEK_SET);
    size_t n = fread(data, 1, count, fp_);
    position_ += n;
    return static_cast<int64_t>(n);
  }

  int64_t ReadAt(int64_t offset, void* data, int64_t count) override {
    if (!fp_) return -1;
    fseek(fp_, offset, SEEK_SET);
    size_t n = fread(data, 1, count, fp_);
    // ReadAt should not change position_
    return static_cast<int64_t>(n);
  }

  void Seek(int64_t position) override { position_ = position; }

  int64_t GetSize() override { return file_size_; }

  const std::string& filename() const override { return filename_; }

  bool is_open() const { return fp_ != nullptr; }

 private:
  std::string filename_;
  FILE* fp_ = nullptr;
  int64_t position_ = 0;
  int64_t file_size_ = 0;
};

class IndexFileReaderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    test_file_ = "/tmp/tenann_test_index_file_reader.bin";
    // Write test data to file
    test_data_.resize(1024);
    for (size_t i = 0; i < test_data_.size(); i++) {
      test_data_[i] = static_cast<uint8_t>(i & 0xFF);
    }
    FILE* fp = fopen(test_file_.c_str(), "wb");
    ASSERT_NE(fp, nullptr);
    fwrite(test_data_.data(), 1, test_data_.size(), fp);
    fclose(fp);
  }

  void TearDown() override { std::remove(test_file_.c_str()); }

  std::string test_file_;
  std::vector<uint8_t> test_data_;
};

TEST_F(IndexFileReaderTest, LocalFileReader_Open) {
  auto reader = std::make_shared<LocalIndexFileReader>(test_file_);
  EXPECT_TRUE(reader->is_open());
  EXPECT_EQ(reader->GetSize(), 1024);
  EXPECT_EQ(reader->filename(), test_file_);
}

TEST_F(IndexFileReaderTest, LocalFileReader_OpenFail) {
  auto reader = std::make_shared<LocalIndexFileReader>("/tmp/non_existent_file_xxx.bin");
  EXPECT_FALSE(reader->is_open());
  EXPECT_EQ(reader->GetSize(), 0);
}

TEST_F(IndexFileReaderTest, LocalFileReader_SequentialRead) {
  auto reader = std::make_shared<LocalIndexFileReader>(test_file_);
  // Read first 100 bytes
  std::vector<uint8_t> buf(100);
  int64_t n = reader->Read(buf.data(), 100);
  EXPECT_EQ(n, 100);
  EXPECT_EQ(memcmp(buf.data(), test_data_.data(), 100), 0);

  // Read next 200 bytes
  buf.resize(200);
  n = reader->Read(buf.data(), 200);
  EXPECT_EQ(n, 200);
  EXPECT_EQ(memcmp(buf.data(), test_data_.data() + 100, 200), 0);
}

TEST_F(IndexFileReaderTest, LocalFileReader_ReadAt) {
  auto reader = std::make_shared<LocalIndexFileReader>(test_file_);
  std::vector<uint8_t> buf(50);

  // Read at offset 500
  int64_t n = reader->ReadAt(500, buf.data(), 50);
  EXPECT_EQ(n, 50);
  EXPECT_EQ(memcmp(buf.data(), test_data_.data() + 500, 50), 0);

  // ReadAt should not affect sequential position
  // Sequential read should still start from 0
  std::vector<uint8_t> buf2(10);
  n = reader->Read(buf2.data(), 10);
  EXPECT_EQ(n, 10);
  EXPECT_EQ(memcmp(buf2.data(), test_data_.data(), 10), 0);
}

TEST_F(IndexFileReaderTest, LocalFileReader_Seek) {
  auto reader = std::make_shared<LocalIndexFileReader>(test_file_);

  // Seek to offset 512, then read
  reader->Seek(512);
  std::vector<uint8_t> buf(100);
  int64_t n = reader->Read(buf.data(), 100);
  EXPECT_EQ(n, 100);
  EXPECT_EQ(memcmp(buf.data(), test_data_.data() + 512, 100), 0);
}

TEST_F(IndexFileReaderTest, LocalFileReader_ReadPastEnd) {
  auto reader = std::make_shared<LocalIndexFileReader>(test_file_);
  reader->Seek(1000);
  std::vector<uint8_t> buf(100);
  int64_t n = reader->Read(buf.data(), 100);
  // Only 24 bytes remaining
  EXPECT_EQ(n, 24);
}

TEST_F(IndexFileReaderTest, CustomFaissIOReader_Basic) {
  auto file_reader = std::make_shared<LocalIndexFileReader>(test_file_);
  CustomFaissIOReader io_reader(file_reader);

  EXPECT_EQ(io_reader.name, test_file_);
  EXPECT_EQ(io_reader.bytes_read(), 0u);

  // Read 4 items of 8 bytes each (simulating FAISS READ1/READVECTOR)
  std::vector<uint8_t> buf(32);
  size_t nitems = io_reader(buf.data(), 8, 4);
  EXPECT_EQ(nitems, 4u);
  EXPECT_EQ(io_reader.bytes_read(), 32u);
  EXPECT_EQ(memcmp(buf.data(), test_data_.data(), 32), 0);
}

TEST_F(IndexFileReaderTest, CustomFaissIOReader_PartialRead) {
  auto file_reader = std::make_shared<LocalIndexFileReader>(test_file_);
  CustomFaissIOReader io_reader(file_reader);

  // Read 100 items of 1 byte
  std::vector<uint8_t> buf(100);
  size_t nitems = io_reader(buf.data(), 1, 100);
  EXPECT_EQ(nitems, 100u);
  EXPECT_EQ(io_reader.bytes_read(), 100u);

  // Read more
  nitems = io_reader(buf.data(), 1, 100);
  EXPECT_EQ(nitems, 100u);
  EXPECT_EQ(io_reader.bytes_read(), 200u);
  EXPECT_EQ(memcmp(buf.data(), test_data_.data() + 100, 100), 0);
}

TEST_F(IndexFileReaderTest, CustomFaissIOReader_ReadPastEnd) {
  auto file_reader = std::make_shared<LocalIndexFileReader>(test_file_);
  CustomFaissIOReader io_reader(file_reader);

  // Try to read 200 items of 8 bytes (1600 bytes > 1024)
  std::vector<uint8_t> buf(1600, 0);
  size_t nitems = io_reader(buf.data(), 8, 200);
  // Should return 128 complete items (1024 / 8)
  EXPECT_EQ(nitems, 128u);
  EXPECT_EQ(io_reader.bytes_read(), 1024u);
}

TEST_F(IndexFileReaderTest, CustomFaissIOReader_BytesReadTracking) {
  auto file_reader = std::make_shared<LocalIndexFileReader>(test_file_);
  CustomFaissIOReader io_reader(file_reader);

  std::vector<uint8_t> buf(256);
  io_reader(buf.data(), 1, 50);
  EXPECT_EQ(io_reader.bytes_read(), 50u);

  io_reader(buf.data(), 4, 25);
  EXPECT_EQ(io_reader.bytes_read(), 150u);  // 50 + 100

  io_reader(buf.data(), 1, 200);
  EXPECT_EQ(io_reader.bytes_read(), 350u);  // 150 + 200
}

}  // namespace tenann
