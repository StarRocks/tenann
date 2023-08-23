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
 * 正常的日志输出有三个级别：DEBUG、INFO、WARNING。
 * 每种日志都会向std::cerr打印一条信息。
 *
 * 目前的实现不支持日志级别开关，对于全部日志都会无条件输出，后期会补全相关功能。
 * 我们在前期开发中，先尽量少地使用日志，避免日志输出膨胀。
 */
void LogExample() {
  TNN_LOG(DEBUG) << "my debug log";
  TNN_LOG(WARNING) << "my warning log";
  TNN_LOG(INFO) << "my info log";
}

/**
 * 对于错误处理，我们将错误分为两类：
 *    - 可恢复的，对应Error类
 *   - 不可恢复的，对应FatalError类
 *
 * 对于可恢复错误，可以使用LOG(ERROR)以及内置的CHECK和DCHEK系列宏，来记录日志并抛出异常。
 */
void RecoverableErrorExample() {
  int a = 1;
  // 使用LOG(ERROR)会自动抛出Error类型的异常
  try {
    TNN_LOG(ERROR) << "LOG(ERROR) example";
  } catch (tenann::Error& e) {
    std::cerr << "Recover from error 1\n";
  }

  // 使用CHECK宏检查参数，条件不满足会自动抛出Error：
  try {
    TNN_CHECK_GT(a, 100) << "CHECK example";
  } catch (tenann::Error& e) {
    std::cerr << "Recover from error 2\n";
  }

  // 使用DCHECK宏检查参数，条件不满足会自动抛出Error：
  // 和assert类似，DCHECK系列宏仅在DEBUG模式有效，RELEASE模式中会被优化掉。
  // 所以对于可能性比较低的错误，建议用DCHECK来检查，避免正式发布版本中进行检查的额外开销
  try {
    TNN_DCHECK(a > 100) << "DCHECK example";
  } catch (tenann::Error& e) {
    std::cerr << "Recover from error 3\n";
  }
}

/**
 * 对于我们自己内部逻辑导致的不可恢复错误，
 * 可以使用LOG(FATAL)，或者内置的ICHECK系列宏来记录日志并抛出异常。
 */
void FatalErrorExample() {
  int a = 1;

  try {
    TNN_LOG(FATAL) << "fatal error";
  } catch (tenann::FatalError& e) {
    std::cout << "we should let it crash instead of catching a fatal error\n";
  }

  try {
    TNN_ICHECK(a == 100);
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