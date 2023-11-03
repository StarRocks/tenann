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

#include <filesystem>

namespace tenann {

inline bool FileOrDirExists(const std::string& file_path, bool is_dir) {
  std::filesystem::path path(file_path);
  auto exists = std::filesystem::exists(file_path);
  auto path_is_dir = std::filesystem::is_directory(file_path);
  if (is_dir) {
    return exists && path_is_dir;
  } else {
    return exists && !path_is_dir;
  }
}

inline bool FileExists(const std::string& file_path) { return FileOrDirExists(file_path, false); }

inline bool DirExists(const std::string& file_path) { return FileOrDirExists(file_path, true); }

}  // namespace tenann