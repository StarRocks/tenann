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

#include "tenann/common/logging.h"

namespace tenann {

int T_MIN_LOG_LEVEL = T_LOG_LEVEL_INFO;
int T_V_LOG_LEVEL = VERBOSE_CRITICAL;

std::string Backtrace() { return ""; }

namespace detail {

const char* ::tenann::detail::LogMessage::level_strings_[] = {
    ": Debug: ",    // TNN_LOG_LEVEL_DEBUG
    ": INFO: ",     // TNN_LOG_LEVEL_INFO
    ": Warning: ",  // TNN_LOG_LEVEL_WARNING
    ": Error: ",    // TNN_LOG_LEVEL_ERROR
};

LogFatal::Entry& LogFatal::GetEntry() {
  static thread_local LogFatal::Entry result;
  return result;
}

LogError::Entry& LogError::GetEntry() {
  static thread_local LogError::Entry result;
  return result;
}

}  // namespace detail

void SetLogLevel(int level) { T_MIN_LOG_LEVEL = level; }
void SetVLogLevel(int level) { T_V_LOG_LEVEL = level; }

}  // namespace tenann