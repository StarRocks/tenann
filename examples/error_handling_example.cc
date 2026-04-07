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

/**
 * Normal log output has three levels: DEBUG, INFO, WARNING.
 * Each log level prints a message to std::cerr.
 *
 * The current implementation does not support log level filtering; all logs are
 * output unconditionally. This functionality will be added in the future.
 * During early development, minimize log usage to avoid log output bloat.
 */
void LogExample() {
  T_LOG(DEBUG) << "my debug log";
  T_LOG(WARNING) << "my warning log";
  T_LOG(INFO) << "my info log";
}

/**
 * For error handling, errors are divided into two categories:
 *   - Recoverable errors, corresponding to the Error class
 *   - Unrecoverable errors, corresponding to the FatalError class
 *
 * For recoverable errors, use LOG(ERROR) and the built-in CHECK and DCHECK macro
 * series to log and throw exceptions.
 */
void RecoverableErrorExample() {
  int a = 1;
  // Using LOG(ERROR) automatically throws an Error exception
  try {
    T_LOG(ERROR) << "LOG(ERROR) example";
  } catch (tenann::Error& e) {
    std::cerr << "Recover from error 1\n";
  }

  // Using CHECK macros to validate parameters; throws Error if condition is not met:
  try {
    T_CHECK_GT(a, 100) << "CHECK example";
  } catch (tenann::Error& e) {
    std::cerr << "Recover from error 2\n";
  }

  // Using DCHECK macros to validate parameters; throws Error if condition is not met:
  // Similar to assert, DCHECK macros are only active in DEBUG mode and optimized away in RELEASE.
  // For low-probability errors, use DCHECK to avoid the overhead of checks in release builds.
  try {
    T_DCHECK(a > 100) << "DCHECK example";
  } catch (tenann::Error& e) {
    std::cerr << "Recover from error 3\n";
  }
}

/**
 * For unrecoverable errors caused by internal logic,
 * use LOG(FATAL) or the built-in ICHECK macro series to log and throw exceptions.
 */
void FatalErrorExample() {
  int a = 1;

  try {
    T_LOG(FATAL) << "fatal error";
  } catch (tenann::FatalError& e) {
    std::cout << "we should let it crash instead of catching a fatal error\n";
  }

  try {
    T_ICHECK(a == 100);
  } catch (tenann::FatalError& e) {
    std::cout << "we should let it crash instead of catching a fatal error\n";
  }
}

int main() {
  using namespace tenann;

  std::cerr << "---------------- LogExample ----------------\n";
  LogExample();
  std::cerr << "\n";

  std::cerr << "---------------- RecoverableErrorExample ----------------\n";
  RecoverableErrorExample();
  std::cerr << "\n";

  std::cerr << "---------------- FatalErrorExample ----------------\n";
  FatalErrorExample();
  std::cerr << "\n";

  return 0;
}
