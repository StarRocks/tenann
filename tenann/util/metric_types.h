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

#include <cstdint>

namespace tenann {

struct TUnit {
  enum type {
    UNIT = 0,
    UNIT_PER_SECOND = 1,
    // CPU_TICKS = 2, // CPU_TICKS is not supported yet
    BYTES = 3,
    BYTES_PER_SECOND = 4,
    TIME_NS = 5,
    DOUBLE_VALUE = 6,
    NONE = 7,
    TIME_MS = 8,
    TIME_S = 9
  };
};

struct TMetricKind {
  enum type { GAUGE = 0, COUNTER = 1, PROPERTY = 2, STATS = 3, SET = 4, HISTOGRAM = 5 };
};

struct TCounterAggregateType {
  enum type { SUM = 0, AVG = 1 };
};

struct TCounterMergeType {
  enum type { MERGE_ALL = 0, SKIP_ALL = 1, SKIP_FIRST_MERGE = 2, SKIP_SECOND_MERGE = 3 };
};

struct TCounterStrategy {
  TCounterAggregateType::type aggregate_type;
  TCounterMergeType::type merge_type;
  int64_t display_threshold;
};

}  // namespace tenann
