#!/usr/bin/env bash
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# get latest build dir
LATEST_BUILD_DIR=$(ls -td build_* 2>/dev/null | head -n 1)

# Call build.sh with bash, not sh: it uses [[ ]], =~ and $OSTYPE, none of which a
# POSIX sh such as dash provides.
if [ -z "${LATEST_BUILD_DIR}" ]; then
  bash build.sh --with-tests
  LATEST_BUILD_DIR=$(ls -td build_* 2>/dev/null | head -n 1)
else
  DIR_SUFFIX=${LATEST_BUILD_DIR#build_}
  rm -f "${LATEST_BUILD_DIR}/test/tenann_test"
  BUILD_TYPE=${DIR_SUFFIX} bash build.sh --with-tests
fi

# Stop here when the build failed. The branch above deletes the test binary before
# rebuilding, so continuing would report a missing test rather than the build error
# that actually caused it.
if [ -z "${LATEST_BUILD_DIR}" ] || [ ! -x "${LATEST_BUILD_DIR}/test/tenann_test" ]; then
  echo "ERROR: tenann_test was not built; see the build output above."
  exit 1
fi

# TODO: enable after resolving Python environment issues
# python3.6 -m unittest discover -s python_bindings -p "test_*.py"

# Run ctest directly instead of `make test`. build.sh generates a Ninja build, so the
# build directory holds no Makefile; `make test` there matched the build directory's
# own test/ subdirectory, printed "Nothing to be done for 'test'" and exited 0
# without running a single case.
(cd "${LATEST_BUILD_DIR}" && env CTEST_OUTPUT_ON_FAILURE=1 ctest) || exit 1

# make coverage report: MAKE_COVERAGE=1 bash run_ut.sh
if [ "$MAKE_COVERAGE" == "1" ]; then
  cmake --build "${LATEST_BUILD_DIR}" --target coverage
fi
