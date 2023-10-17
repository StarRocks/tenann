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

// Define macros for updating counters.  The macros make it very easy to disable
// all counters at compile time.  Set this to 0 to remove counters.  This is useful
// to do to make sure the counters aren't affecting the system.

#pragma once

#include "tenann/util/runtime_profile.h"

#define T_ENABLE_COUNTERS 1

// Some macro magic to generate unique ids using __COUNTER__
#define T_CONCAT_IMPL(x, y) x##y
#define T_MACRO_CONCAT(x, y) T_CONCAT_IMPL(x, y)

#if T_ENABLE_COUNTERS
#define T_ADD_COUNTER(profile, name, type) \
  (profile) == nullptr                     \
      ? nullptr                            \
      : (profile)->add_counter(name, type, tenann::RuntimeProfile::Counter::create_strategy(type))

#define T_ADD_COUNTER_SKIP_MERGE(profile, name, type, merge_type) \
  (profile) == nullptr                                            \
      ? nullptr                                                   \
      : (profile)->add_counter(name, type,                        \
                               tenann::RuntimeProfile::Counter::create_strategy(type, merge_type))

#define T_ADD_TIMER(profile, name)        \
  (profile) == nullptr                    \
      ? nullptr                           \
      : (profile)->add_counter(           \
            name, tenann::TUnit::TIME_NS, \
            tenann::RuntimeProfile::Counter::create_strategy(tenann::TUnit::TIME_NS))

#define T_ADD_CHILD_COUNTER(profile, name, type, parent) \
  (profile) == nullptr                                   \
      ? nullptr                                          \
      : (profile)->add_child_counter(                    \
            name, type, tenann::RuntimeProfile::Counter::create_strategy(type), parent)

#define T_ADD_CHILD_COUNTER_SKIP_MERGE(profile, name, type, merge_type, parent)             \
  (profile) == nullptr                                                                      \
      ? nullptr                                                                             \
      : (profile)->add_child_counter(                                                       \
            name, type, tenann::RuntimeProfile::Counter::create_strategy(type, merge_type), \
            parent)

#define T_ADD_CHILD_TIMER_THRESHOLD(profile, name, parent, threshold)                     \
  (profile) == nullptr                                                                    \
      ? nullptr                                                                           \
      : (profile)->add_child_counter(                                                     \
            name, tenann::TUnit::TIME_NS,                                                 \
            tenann::RuntimeProfile::Counter::create_strategy(                             \
                tenann::TUnit::TIME_NS, tenann::TCounterMergeType::MERGE_ALL, threshold), \
            parent)

#define T_ADD_CHILD_TIMER(profile, name, parent) \
  (profile) == nullptr                           \
      ? nullptr                                  \
      : (profile)->add_child_counter(            \
            name, tenann::TUnit::TIME_NS,        \
            tenann::RuntimeProfile::Counter::create_strategy(TUnit::TIME_NS), parent)

#define T_SCOPED_TIMER(c) \
  tenann::ScopedTimer<tenann::MonotonicStopWatch> T_MACRO_CONCAT(T_SCOPED_TIMER, __COUNTER__)(c)
#define T_CANCEL_SAFE_SCOPED_TIMER(c, is_cancelled)                                            \
  tenann::ScopedTimer<tenann::MonotonicStopWatch> T_MACRO_CONCAT(T_SCOPED_TIMER, __COUNTER__)( \
      c, is_cancelled)
#define T_SCOPED_RAW_TIMER(c)                                                           \
  tenann::ScopedRawTimer<tenann::MonotonicStopWatch> T_MACRO_CONCAT(T_SCOPED_RAW_TIMER, \
                                                                    __COUNTER__)(c)
#define T_COUNTER_UPDATE(c, v) \
  if ((c) != nullptr) {        \
    (c)->update(v);            \
  }
#define T_OUNTER_SET(c, v) \
  if ((c) != nullptr) {    \
    (c)->set(v);           \
  }
// this is only used for HighWaterMarkCounter
#define T_COUNTER_ADD(c, v) \
  if ((c) != nullptr) {     \
    (c)->add(v);            \
  }
#define T_ADD_THREAD_COUNTERS(profile, prefix) \
  if ((profile) != nullptr) {                  \
    (profile)->add_thread_counters(prefix);    \
  }

#else
#define T_ADD_COUNTER(profile, name, type) NULL
#define T_ADD_TIMER(profile, name) NULL
#define T_SCOPED_TIMER(c)
#define T_SCOPED_RAW_TIMER(c)
#define T_COUNTER_UPDATE(c, v)
#define T_COUNTER_SET(c, v)
#define T_COUNTER_ADD(c, v)
#define T_ADD_THREADCOUNTERS(profile, prefix) NULL
#define T_SCOPED_THREAD_COUNTER_MEASUREMENT(c)
#endif