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

// This file is based on code available under the Apache License 2.0 here:
//     https://github.com/apache/tvm/blob/main/include/tvm/runtime/logging.h

#pragma once

#include "tenann/common/error.h"
#include "tenann/common/macros.h"

namespace tenann {

namespace detail {
/*!
 * \brief Class to accumulate an error message and throw it. Do not use
 * directly, instead use LOG(FATAL).
 * \note The `LogFatal` class is designed to be an empty class to reduce stack size usage.
 * To play this trick, we use the thread-local storage to store its internal data.
 */
class LogFatal {
 public:
  LogFatal(const char* file, int lineno) { GetEntry().Init(file, lineno); }
#ifdef _MSC_VER
#pragma disagnostic push
#pragma warning(disable : 4722)
#endif
  [[noreturn]] ~LogFatal() T_THROW_EXCEPTION { GetEntry().Finalize(); }
#ifdef _MSC_VER
#pragma disagnostic pop
#endif
  std::ostringstream& stream() { return GetEntry().stream_; }

 private:
  struct Entry {
    void Init(const char* file, int lineno) {
      this->stream_.str("");
      this->file_ = file;
      this->lineno_ = lineno;
    }
    [[noreturn]] T_NO_INLINE Error Finalize() {
      FatalError error(file_, lineno_, stream_.str());
      std::cerr << error.what();
      throw error;
    }
    std::ostringstream stream_;
    std::string file_;
    int lineno_;
  };

  T_NO_INLINE static Entry& GetEntry();
};

/*!
 * \brief Class to accumulate an error message and throw it. Do not use
 * directly, instead use LOG(ERROR).
 * \note The `LogError` class is designed to be an empty class to reduce stack size usage.
 * To play this trick, we use the thread-local storage to store its internal data.
 */
class LogError {
 public:
  LogError(const char* file, int lineno) { GetEntry().Init(file, lineno); }
#ifdef _MSC_VER
#pragma disagnostic push
#pragma warning(disable : 4722)
#endif
  [[noreturn]] ~LogError() T_THROW_EXCEPTION { GetEntry().Finalize(); }
#ifdef _MSC_VER
#pragma disagnostic pop
#endif
  std::ostringstream& stream() { return GetEntry().stream_; }

 private:
  struct Entry {
    void Init(const char* file, int lineno) {
      this->stream_.str("");
      this->file_ = file;
      this->lineno_ = lineno;
    }
    [[noreturn]] T_NO_INLINE Error Finalize() {
      Error error(file_, lineno_, stream_.str());
      std::cerr << error.what();
      throw error;
    }
    std::ostringstream stream_;
    std::string file_;
    int lineno_;
  };

  T_NO_INLINE static Entry& GetEntry();
};

/*!
 * \brief Class to accumulate an log message. Do not use directly, instead use
 * LOG(INFO), LOG(WARNING), LOG(ERROR).
 */
class LogMessage {
 public:
  LogMessage(const std::string& file, int lineno, int level) {
    std::time_t t = std::time(nullptr);
    stream_ << "[" << std::put_time(std::localtime(&t), "%Y-%m-%d %H:%M:%S") << "] " << file << ":"
            << lineno << level_strings_[level];
  }
  T_NO_INLINE ~LogMessage() { std::cerr << stream_.str() << std::endl; }
  std::ostringstream& stream() { return stream_; }

 private:
  std::ostringstream stream_;
  static const char* level_strings_[];
};

// Below is from dmlc-core
// This class is used to explicitly ignore values in the conditional
// logging macros.  This avoids compiler warnings like "value computed
// is not used" and "statement has no effect".
class LogMessageVoidify {
 public:
  LogMessageVoidify() {}
  // This has to be an operator with a precedence lower than << but
  // higher than "?:". See its usage.
  void operator&(std::ostream&) {}
};

template <typename X, typename Y>
std::unique_ptr<std::string> LogCheckFormat(const X& x, const Y& y) {
  std::ostringstream os;
  os << " (" << x << " vs. " << y << ") ";  // CHECK_XX(x, y) requires x and y can be serialized
                                            // to string. Use CHECK(x OP y) otherwise.
  return std::make_unique<std::string>(os.str());
}

// Inline _Pragma in macros does not work reliably on old version of MSVC and
// GCC. We wrap all comparisons in a function so that we can use #pragma to
// silence bad comparison warnings.
#define T_CHECK_FUNC(name, op)                                                          \
  template <typename X, typename Y>                                                     \
  T_ALWAYS_INLINE std::unique_ptr<std::string> LogCheck##name(const X& x, const Y& y) { \
    if (x op y) return nullptr;                                                         \
    return LogCheckFormat(x, y);                                                        \
  }                                                                                     \
  T_ALWAYS_INLINE std::unique_ptr<std::string> LogCheck##name(int x, int y) {           \
    return LogCheck##name<int, int>(x, y);                                              \
  }

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
T_CHECK_FUNC(_LT, <)
T_CHECK_FUNC(_GT, >)
T_CHECK_FUNC(_LE, <=)
T_CHECK_FUNC(_GE, >=)
T_CHECK_FUNC(_EQ, ==)
T_CHECK_FUNC(_NE, !=)
#pragma GCC diagnostic pop

}  // namespace detail

#define T_LOG_LEVEL_DEBUG 0
#define T_LOG_LEVEL_INFO 1
#define T_LOG_LEVEL_WARNING 2
#define T_LOG_LEVEL_ERROR 3
#define T_LOG_LEVEL_FATAL 4
#define T_LOG(level) T_LOG_##level
#define T_LOG_DEBUG ::tenann::detail::LogMessage(__FILE__, __LINE__, T_LOG_LEVEL_DEBUG).stream()
#define T_LOG_FATAL ::tenann::detail::LogFatal(__FILE__, __LINE__).stream()
#define T_LOG_INFO ::tenann::detail::LogMessage(__FILE__, __LINE__, T_LOG_LEVEL_INFO).stream()
#define T_LOG_ERROR ::tenann::detail::LogError(__FILE__, __LINE__).stream()
#define T_LOG_WARNING ::tenann::detail::LogMessage(__FILE__, __LINE__, T_LOG_LEVEL_WARNING).stream()

#define T_CHECK_BINARY_OP(name, op, x, y)                          \
  if (auto __t__log__err = ::tenann::detail::LogCheck##name(x, y)) \
  ::tenann::detail::LogError(__FILE__, __LINE__).stream()          \
      << "Check failed: " << #x " " #op " " #y << *__t__log__err << ": "

#define T_CHECK_LT(x, y) T_CHECK_BINARY_OP(_LT, <, x, y)
#define T_CHECK_GT(x, y) T_CHECK_BINARY_OP(_GT, >, x, y)
#define T_CHECK_LE(x, y) T_CHECK_BINARY_OP(_LE, <=, x, y)
#define T_CHECK_GE(x, y) T_CHECK_BINARY_OP(_GE, >=, x, y)
#define T_CHECK_EQ(x, y) T_CHECK_BINARY_OP(_EQ, ==, x, y)
#define T_CHECK_NE(x, y) T_CHECK_BINARY_OP(_NE, !=, x, y)
#define T_CHECK_NOTNULL(x)                                                                    \
  ((x) == nullptr                                                                             \
   ? ::tenann::detail::LogError(__FILE__, __LINE__).stream() << "Check not null: " #x << ' ', \
   (x) : (x))  // NOLINT(*)

#define T_CHECK(x) \
  if (!(x))        \
  ::tenann::detail::LogError(__FILE__, __LINE__).stream() << "Check failed: (" #x << ") is false: "

#define T_LOG_IF(severity, condition) \
  !(condition) ? (void)0 : ::tenann::detail::LogMessageVoidify() & T_LOG(severity)

#ifndef NDEBUG
#define T_DCHECK(x) T_CHECK(x)
#define T_DCHECK_LT(x, y) T_CHECK((x) < (y))
#define T_DCHECK_GT(x, y) T_CHECK((x) > (y))
#define T_DCHECK_LE(x, y) T_CHECK((x) <= (y))
#define T_DCHECK_GE(x, y) T_CHECK((x) >= (y))
#define T_DCHECK_EQ(x, y) T_CHECK((x) == (y))
#define T_DCHECK_NE(x, y) T_CHECK((x) != (y))
#define T_DCHECK_NOTNULL(x) T_CHECK_NOTNULL((x))
#else
#define T_DCHECK(x) \
  while (false) T_CHECK(x)
#define T_DCHECK_LT(x, y) \
  while (false) T_CHECK((x) < (y))
#define T_DCHECK_GT(x, y) \
  while (false) T_CHECK((x) > (y))
#define T_DCHECK_LE(x, y) \
  while (false) T_CHECK((x) <= (y))
#define T_DCHECK_GE(x, y) \
  while (false) T_CHECK((x) >= (y))
#define T_DCHECK_EQ(x, y) \
  while (false) T_CHECK((x) == (y))
#define T_DCHECK_NE(x, y) \
  while (false) T_CHECK((x) != (y))
#define T_DCHECK_NOTNULL(x) (x)
#endif

#define T_ICHECK_INDENT "  "

#define T_ICHECK(x)                                       \
  if (!(x))                                               \
  ::tenann::detail::LogFatal(__FILE__, __LINE__).stream() \
      << "FatalError: Check failed: (" #x << ") is false: "

#define T_ICHECK_LT(x, y) T_ICHECK_BINARY_OP(_LT, <, x, y)
#define T_ICHECK_GT(x, y) T_ICHECK_BINARY_OP(_GT, >, x, y)
#define T_ICHECK_LE(x, y) T_ICHECK_BINARY_OP(_LE, <=, x, y)
#define T_ICHECK_GE(x, y) T_ICHECK_BINARY_OP(_GE, >=, x, y)
#define T_ICHECK_EQ(x, y) T_ICHECK_BINARY_OP(_EQ, ==, x, y)
#define T_ICHECK_NE(x, y) T_ICHECK_BINARY_OP(_NE, !=, x, y)
#define T_ICHECK_NOTNULL(x)                                                 \
  ((x) == nullptr ? ::tenann::detail::LogFatal(__FILE__, __LINE__).stream() \
                        << "FatalError: Check not null: " #x << ' ',        \
   (x) : (x))  // NOLINT(*)
}  // namespace tenann