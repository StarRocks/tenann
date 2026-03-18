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
#include "tenann/index/faiss_io_reader_adapter.h"
#include "tenann/index/faiss_io_writer_adapter.h"
#include "tenann/store/index_file_reader.h"
#include "tenann/store/index_file_writer.h"

namespace tenann {

/// A local-file-based IndexFileWriter for testing purposes.
/// Wraps POSIX FILE* to simulate writing to any file system.
class LocalIndexFileWriter : public IndexFileWriter {
 public:
  explicit LocalIndexFileWriter(const std::string& path) : filename_(path) {
    fp_ = fopen(path.c_str(), "wb");
  }

  ~LocalIndexFileWriter() override {
    if (fp_) fclose(fp_);
  }

  int64_t Write(const void* data, int64_t count) override {
    if (!fp_) return -1;
    size_t n = fwrite(data, 1, count, fp_);
    return static_cast<int64_t>(n);
  }

  void Flush() override {
    if (fp_) fflush(fp_);
  }

  void Close() override {
    if (fp_) {
      fclose(fp_);
      fp_ = nullptr;
    }
  }

  const std::string& filename() const override { return filename_; }

  bool is_open() const { return fp_ != nullptr; }

 private:
  std::string filename_;
  FILE* fp_ = nullptr;
};

/// A local-file-based IndexFileReader for verifying written data.
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
    return static_cast<int64_t>(fread(data, 1, count, fp_));
  }

  void Seek(int64_t position) override { position_ = position; }
  int64_t GetSize() override { return file_size_; }
  const std::string& filename() const override { return filename_; }

 private:
  std::string filename_;
  FILE* fp_ = nullptr;
  int64_t position_ = 0;
  int64_t file_size_ = 0;
};

class IndexFileWriterTest : public ::testing::Test {
 protected:
  void SetUp() override {
    test_file_ = "/tmp/tenann_test_index_file_writer.bin";
    // Prepare test data
    test_data_.resize(1024);
    for (size_t i = 0; i < test_data_.size(); i++) {
      test_data_[i] = static_cast<uint8_t>(i & 0xFF);
    }
  }

  void TearDown() override { std::remove(test_file_.c_str()); }

  std::string test_file_;
  std::vector<uint8_t> test_data_;
};

TEST_F(IndexFileWriterTest, LocalFileWriter_Open) {
  auto writer = std::make_shared<LocalIndexFileWriter>(test_file_);
  EXPECT_TRUE(writer->is_open());
  EXPECT_EQ(writer->filename(), test_file_);
}

TEST_F(IndexFileWriterTest, LocalFileWriter_OpenFail) {
  auto writer = std::make_shared<LocalIndexFileWriter>("/nonexistent_dir/file.bin");
  EXPECT_FALSE(writer->is_open());
}

TEST_F(IndexFileWriterTest, LocalFileWriter_Write) {
  {
    auto writer = std::make_shared<LocalIndexFileWriter>(test_file_);
    int64_t n = writer->Write(test_data_.data(), test_data_.size());
    EXPECT_EQ(n, 1024);
    writer->Close();
  }

  // Verify written content
  std::vector<uint8_t> buf(1024);
  FILE* fp = fopen(test_file_.c_str(), "rb");
  ASSERT_NE(fp, nullptr);
  size_t read = fread(buf.data(), 1, 1024, fp);
  fclose(fp);
  EXPECT_EQ(read, 1024u);
  EXPECT_EQ(memcmp(buf.data(), test_data_.data(), 1024), 0);
}

TEST_F(IndexFileWriterTest, LocalFileWriter_MultipleWrites) {
  {
    auto writer = std::make_shared<LocalIndexFileWriter>(test_file_);
    // Write in chunks
    int64_t n1 = writer->Write(test_data_.data(), 100);
    int64_t n2 = writer->Write(test_data_.data() + 100, 200);
    int64_t n3 = writer->Write(test_data_.data() + 300, 724);
    EXPECT_EQ(n1, 100);
    EXPECT_EQ(n2, 200);
    EXPECT_EQ(n3, 724);
    writer->Close();
  }

  // Verify written content
  std::vector<uint8_t> buf(1024);
  FILE* fp = fopen(test_file_.c_str(), "rb");
  ASSERT_NE(fp, nullptr);
  size_t nread = fread(buf.data(), 1, 1024, fp);
  fclose(fp);
  EXPECT_EQ(nread, 1024u);
  EXPECT_EQ(memcmp(buf.data(), test_data_.data(), 1024), 0);
}

TEST_F(IndexFileWriterTest, LocalFileWriter_Flush) {
  auto writer = std::make_shared<LocalIndexFileWriter>(test_file_);
  writer->Write(test_data_.data(), 512);
  writer->Flush();

  // File should be readable after flush (data flushed to disk)
  FILE* fp = fopen(test_file_.c_str(), "rb");
  ASSERT_NE(fp, nullptr);
  fseek(fp, 0, SEEK_END);
  long size = ftell(fp);
  fclose(fp);
  EXPECT_EQ(size, 512);
}

TEST_F(IndexFileWriterTest, LocalFileWriter_Close) {
  auto writer = std::make_shared<LocalIndexFileWriter>(test_file_);
  EXPECT_TRUE(writer->is_open());
  writer->Write(test_data_.data(), 100);
  writer->Close();
  EXPECT_FALSE(writer->is_open());

  // Write after close should return -1
  int64_t n = writer->Write(test_data_.data(), 100);
  EXPECT_EQ(n, -1);
}

TEST_F(IndexFileWriterTest, FaissIOWriterAdapter_Basic) {
  auto file_writer = std::make_shared<LocalIndexFileWriter>(test_file_);
  FaissIOWriterAdapter adapter(file_writer);

  EXPECT_EQ(adapter.name, test_file_);

  // Write 4 items of 8 bytes each (simulating FAISS WRITE1/WRITEVECTOR)
  size_t nitems = adapter(test_data_.data(), 8, 4);
  EXPECT_EQ(nitems, 4u);
  file_writer->Close();

  // Verify written content
  std::vector<uint8_t> buf(32);
  FILE* fp = fopen(test_file_.c_str(), "rb");
  ASSERT_NE(fp, nullptr);
  size_t nread = fread(buf.data(), 1, 32, fp);
  fclose(fp);
  EXPECT_EQ(nread, 32u);
  EXPECT_EQ(memcmp(buf.data(), test_data_.data(), 32), 0);
}

TEST_F(IndexFileWriterTest, FaissIOWriterAdapter_MultipleWrites) {
  auto file_writer = std::make_shared<LocalIndexFileWriter>(test_file_);
  FaissIOWriterAdapter adapter(file_writer);

  // Write 100 items of 1 byte
  size_t n1 = adapter(test_data_.data(), 1, 100);
  EXPECT_EQ(n1, 100u);

  // Write 25 items of 4 bytes
  size_t n2 = adapter(test_data_.data() + 100, 4, 25);
  EXPECT_EQ(n2, 25u);

  // Write 50 items of 1 byte
  size_t n3 = adapter(test_data_.data() + 200, 1, 50);
  EXPECT_EQ(n3, 50u);

  file_writer->Close();

  // Verify: total 250 bytes written
  std::vector<uint8_t> buf(250);
  FILE* fp = fopen(test_file_.c_str(), "rb");
  ASSERT_NE(fp, nullptr);
  size_t nread = fread(buf.data(), 1, 250, fp);
  fclose(fp);
  EXPECT_EQ(nread, 250u);
  EXPECT_EQ(memcmp(buf.data(), test_data_.data(), 250), 0);
}

TEST_F(IndexFileWriterTest, FaissIOWriterAdapter_WriteThenRead) {
  // Write via FaissIOWriterAdapter
  {
    auto file_writer = std::make_shared<LocalIndexFileWriter>(test_file_);
    FaissIOWriterAdapter writer_adapter(file_writer);

    writer_adapter(test_data_.data(), 1, test_data_.size());
    file_writer->Close();
  }

  // Read back via FaissIOReaderAdapter and verify round-trip
  {
    auto file_reader = std::make_shared<LocalIndexFileReader>(test_file_);
    EXPECT_EQ(file_reader->GetSize(), 1024);

    FaissIOReaderAdapter reader_adapter(file_reader);
    std::vector<uint8_t> buf(1024);
    size_t nitems = reader_adapter(buf.data(), 1, 1024);
    EXPECT_EQ(nitems, 1024u);
    EXPECT_EQ(memcmp(buf.data(), test_data_.data(), 1024), 0);
  }
}

TEST_F(IndexFileWriterTest, FaissIOWriterAdapter_LargeWrite) {
  // Write large data (64KB)
  std::vector<uint8_t> large_data(65536);
  for (size_t i = 0; i < large_data.size(); i++) {
    large_data[i] = static_cast<uint8_t>((i * 7 + 13) & 0xFF);
  }

  {
    auto file_writer = std::make_shared<LocalIndexFileWriter>(test_file_);
    FaissIOWriterAdapter adapter(file_writer);
    // Write as 8192 items of 8 bytes
    size_t nitems = adapter(large_data.data(), 8, 8192);
    EXPECT_EQ(nitems, 8192u);
    file_writer->Close();
  }

  // Verify
  {
    auto file_reader = std::make_shared<LocalIndexFileReader>(test_file_);
    EXPECT_EQ(file_reader->GetSize(), 65536);
    std::vector<uint8_t> buf(65536);
    file_reader->Read(buf.data(), 65536);
    EXPECT_EQ(memcmp(buf.data(), large_data.data(), 65536), 0);
  }
}

}  // namespace tenann
