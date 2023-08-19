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
#include "tenann/index/index_reader.h"
#include "tenann/index/index_writer.h"
#include "tenann/scanner/index_scanner.h"
#include "tenann/store/index_meta.h"
#include "tenann/streamer/index_streamer.h"

namespace tenann {

class IndexFactory {
  virtual IndexReader* CreateReaderFromMeta(const IndexMeta& meta);
  virtual IndexWriter* CreateWriterFromMeta(const IndexMeta& meta);
  virtual IndexBuilder* CreateBuilderFromMeta(const IndexMeta& meta);
  virtual IndexStreamer* CreateStreamerFromMeta(const IndexMeta& meta);
  virtual IndexScanner* CreateScannerFromMeta(const IndexMeta& meta);
};

}  // namespace tenann