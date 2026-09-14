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

#include <string>

namespace tenann {

/// Name of the instruction set the vector-search kernels are dispatching to on this
/// machine, for example "AVX512_SPR", "AVX512", "AVX2", "ARM_SVE", "ARM_NEON" or
/// "NONE". The choice is made once, by probing the CPU when the library loads.
///
/// Because the kernels are selected at run time, neither the build flags nor the
/// library file name tell you which ones are in use; this is the only reliable
/// source. Callers that care which ISA a host ended up on -- a server logging its
/// startup configuration, say -- should report this value, so that a CPU probed
/// differently than expected is visible rather than silent.
std::string SimdLevelName();

}  // namespace tenann
