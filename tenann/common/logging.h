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

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "tenann/common/macros.h"

namespace tenann {

/// @TODO(petri): implement this function with libbacktrace
std::string Backtrace();

/*!
 * \brief Error type for errors from TNN_CHECK and LOG(ERROR). This error
 * contains a backtrace of where it occurred.
 */
class Error : std::exception {
 public:
  /*! \brief Construct an error. Not recommended to use directly. Instead use LOG(FATAL).
   *
   * \param file The file where the error occurred.
   * \param lineno The line number where the error occurred.
   * \param message The error message to display.
   * \param time The time at which the error occurred. This should be in local time.
   * \param backtrace Backtrace from when the error occurred.
   */
  Error(std::string file, int lineno, std::string message, std::time_t time = std::time(nullptr),
        std::string backtrace = Backtrace())
      : file_(file), lineno_(lineno), message_(message), time_(time), backtrace_(backtrace) {
    std::ostringstream s;
    // XXX: Do not change this format, otherwise all error handling in python will break (because it
    // parses the message to reconstruct the error type).
    // TODO(tkonolige): Convert errors to Objects, so we can avoid the mess of formatting/parsing
    // error messages correctly.

    s << "[" << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S") << "] " << file << ":"
      << lineno << ": Error: " << message << std::endl;
    if (backtrace.size() > 0) {
      s << backtrace << std::endl;
    }
    full_message_ = s.str();
  }
  /*! \return The file in which the error occurred. */
  const std::string& file() const { return file_; }
  /*! \return The message associated with this error. */
  const std::string& message() const { return message_; }
  /*! \return Formatted error message including file, linenumber, backtrace, and message. */
  const std::string& full_message() const { return full_message_; }
  /*! \return The backtrace from where this error occurred. */
  const std::string& backtrace() const { return backtrace_; }
  /*! \return The time at which this error occurred. */
  const std::time_t& time() const { return time_; }
  /*! \return The line number at which this error occurred. */
  int lineno() const { return lineno_; }
  virtual const char* what() const noexcept { return full_message_.c_str(); }

 private:
  std::string file_;
  int lineno_;
  std::string message_;
  std::time_t time_;
  std::string backtrace_;
  std::string full_message_;  // holds the full error string
};

/*!
 * \brief Error type for errors from TNN_ICHECK and LOG(FATAL). This error
 * contains a backtrace of where it occurred.
 */
class FatalError : std::exception {
 public:
  /*! \brief Construct an error. Not recommended to use directly. Instead use LOG(FATAL).
   *
   * \param file The file where the error occurred.
   * \param lineno The line number where the error occurred.
   * \param message The error message to display.
   * \param time The time at which the error occurred. This should be in local time.
   * \param backtrace Backtrace from when the error occurred.
   */
  FatalError(std::string file, int lineno, std::string message,
             std::time_t time = std::time(nullptr), std::string backtrace = Backtrace())
      : file_(file), lineno_(lineno), message_(message), time_(time), backtrace_(backtrace) {
    std::ostringstream s;
    // XXX: Do not change this format, otherwise all error handling in python will break (because it
    // parses the message to reconstruct the error type).
    // TODO(tkonolige): Convert errors to Objects, so we can avoid the mess of formatting/parsing
    // error messages correctly.

    s << "[" << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S") << "] " << file << ":"
      << lineno << ": Fatal: " << message << std::endl;
    if (backtrace.size() > 0) {
      s << backtrace << std::endl;
    }
    full_message_ = s.str();
  }
  /*! \return The file in which the error occurred. */
  const std::string& file() const { return file_; }
  /*! \return The message associated with this error. */
  const std::string& message() const { return message_; }
  /*! \return Formatted error message including file, linenumber, backtrace, and message. */
  const std::string& full_message() const { return full_message_; }
  /*! \return The backtrace from where this error occurred. */
  const std::string& backtrace() const { return backtrace_; }
  /*! \return The time at which this error occurred. */
  const std::time_t& time() const { return time_; }
  /*! \return The line number at which this error occurred. */
  int lineno() const { return lineno_; }
  virtual const char* what() const noexcept { return full_message_.c_str(); }

 private:
  std::string file_;
  int lineno_;
  std::string message_;
  std::time_t time_;
  std::string backtrace_;
  std::string full_message_;  // holds the full error string
};

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
  [[noreturn]] ~LogFatal() TNN_THROW_EXCEPTION { GetEntry().Finalize(); }
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
    [[noreturn]] TNN_NO_INLINE Error Finalize() {
      FatalError error(file_, lineno_, stream_.str());
      std::cerr << error.what();
      throw error;
    }
    std::ostringstream stream_;
    std::string file_;
    int lineno_;
  };

  TNN_NO_INLINE static Entry& GetEntry();
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
  [[noreturn]] ~LogError() TNN_THROW_EXCEPTION { GetEntry().Finalize(); }
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
    [[noreturn]] TNN_NO_INLINE Error Finalize() {
      Error error(file_, lineno_, stream_.str());
      std::cerr << error.what();
      throw error;
    }
    std::ostringstream stream_;
    std::string file_;
    int lineno_;
  };

  TNN_NO_INLINE static Entry& GetEntry();
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
  TNN_NO_INLINE ~LogMessage() { std::cerr << stream_.str() << std::endl; }
  std::ostringstream& stream() { return stream_; }

 private:
  std::ostringstream stream_;
  static const char* level_strings_[];
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
#define TNN_CHECK_FUNC(name, op)                                                          \
  template <typename X, typename Y>                                                       \
  TNN_ALWAYS_INLINE std::unique_ptr<std::string> LogCheck##name(const X& x, const Y& y) { \
    if (x op y) return nullptr;                                                           \
    return LogCheckFormat(x, y);                                                          \
  }                                                                                       \
  TNN_ALWAYS_INLINE std::unique_ptr<std::string> LogCheck##name(int x, int y) {           \
    return LogCheck##name<int, int>(x, y);                                                \
  }

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
TNN_CHECK_FUNC(_LT, <)
TNN_CHECK_FUNC(_GT, >)
TNN_CHECK_FUNC(_LE, <=)
TNN_CHECK_FUNC(_GE, >=)
TNN_CHECK_FUNC(_EQ, ==)
TNN_CHECK_FUNC(_NE, !=)
#pragma GCC diagnostic pop

}  // namespace detail

#define TNN_LOG_LEVEL_DEBUG 0
#define TNN_LOG_LEVEL_INFO 1
#define TNN_LOG_LEVEL_WARNING 2
#define TNN_LOG_LEVEL_ERROR 3
#define TNN_LOG_LEVEL_FATAL 4
#define TNN_LOG(level) TNN_LOG_##level
#define TNN_LOG_DEBUG ::tenann::detail::LogMessage(__FILE__, __LINE__, TNN_LOG_LEVEL_DEBUG).stream()
#define TNN_LOG_FATAL ::tenann::detail::LogFatal(__FILE__, __LINE__).stream()
#define TNN_LOG_INFO ::tenann::detail::LogMessage(__FILE__, __LINE__, TNN_LOG_LEVEL_INFO).stream()
#define TNN_LOG_ERROR ::tenann::detail::LogError(__FILE__, __LINE__).stream()
#define TNN_LOG_WARNING \
  ::tenann::detail::LogMessage(__FILE__, __LINE__, TNN_LOG_LEVEL_WARNING).stream()

#define TNN_CHECK_BINARY_OP(name, op, x, y)                          \
  if (auto __tvm__log__err = ::tenann::detail::LogCheck##name(x, y)) \
  ::tenann::detail::LogError(__FILE__, __LINE__).stream()            \
      << "Check failed: " << #x " " #op " " #y << *__tvm__log__err << ": "

#define TNN_CHECK_LT(x, y) TNN_CHECK_BINARY_OP(_LT, <, x, y)
#define TNN_CHECK_GT(x, y) TNN_CHECK_BINARY_OP(_GT, >, x, y)
#define TNN_CHECK_LE(x, y) TNN_CHECK_BINARY_OP(_LE, <=, x, y)
#define TNN_CHECK_GE(x, y) TNN_CHECK_BINARY_OP(_GE, >=, x, y)
#define TNN_CHECK_EQ(x, y) TNN_CHECK_BINARY_OP(_EQ, ==, x, y)
#define TNN_CHECK_NE(x, y) TNN_CHECK_BINARY_OP(_NE, !=, x, y)
#define TNN_CHECK_NOTNULL(x)                                                                  \
  ((x) == nullptr                                                                             \
   ? ::tenann::detail::LogError(__FILE__, __LINE__).stream() << "Check not null: " #x << ' ', \
   (x) : (x))  // NOLINT(*)

#define TNN_CHECK(x) \
  if (!(x))          \
  ::tenann::detail::LogError(__FILE__, __LINE__).stream() << "Check failed: (" #x << ") is false: "

#ifndef NDEBUG
#define TNN_DCHECK(x) TNN_CHECK(x)
#define TNN_DCHECK_LT(x, y) TNN_CHECK((x) < (y))
#define TNN_DCHECK_GT(x, y) TNN_CHECK((x) > (y))
#define TNN_DCHECK_LE(x, y) TNN_CHECK((x) <= (y))
#define TNN_DCHECK_GE(x, y) TNN_CHECK((x) >= (y))
#define TNN_DCHECK_EQ(x, y) TNN_CHECK((x) == (y))
#define TNN_DCHECK_NE(x, y) TNN_CHECK((x) != (y))
#else
#define TNN_DCHECK(x) \
  while (false) TNN_CHECK(x)
#define TNN_DCHECK_LT(x, y) \
  while (false) TNN_CHECK((x) < (y))
#define TNN_DCHECK_GT(x, y) \
  while (false) TNN_CHECK((x) > (y))
#define TNN_DCHECK_LE(x, y) \
  while (false) TNN_CHECK((x) <= (y))
#define TNN_DCHECK_GE(x, y) \
  while (false) TNN_CHECK((x) >= (y))
#define TNN_DCHECK_EQ(x, y) \
  while (false) TNN_CHECK((x) == (y))
#define TNN_DCHECK_NE(x, y) \
  while (false) TNN_CHECK((x) != (y))
#endif

#define TNN_ICHECK_INDENT "  "

#define TNN_ICHECK(x)                                     \
  if (!(x))                                               \
  ::tenann::detail::LogFatal(__FILE__, __LINE__).stream() \
      << "FatalError: Check failed: (" #x << ") is false: "

#define TNN_ICHECK_LT(x, y) TNN_ICHECK_BINARY_OP(_LT, <, x, y)
#define TNN_ICHECK_GT(x, y) TNN_ICHECK_BINARY_OP(_GT, >, x, y)
#define TNN_ICHECK_LE(x, y) TNN_ICHECK_BINARY_OP(_LE, <=, x, y)
#define TNN_ICHECK_GE(x, y) TNN_ICHECK_BINARY_OP(_GE, >=, x, y)
#define TNN_ICHECK_EQ(x, y) TNN_ICHECK_BINARY_OP(_EQ, ==, x, y)
#define TNN_ICHECK_NE(x, y) TNN_ICHECK_BINARY_OP(_NE, !=, x, y)
#define TNN_ICHECK_NOTNULL(x)                                               \
  ((x) == nullptr ? ::tenann::detail::LogFatal(__FILE__, __LINE__).stream() \
                        << "FatalError: Check not null: " #x << ' ',        \
   (x) : (x))  // NOLINT(*)
}  // namespace tenann