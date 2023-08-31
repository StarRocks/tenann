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
LATEST_BUILD_DIR=$(ls -td build_* | head -n 1)

if [ -z ${LATEST_BUILD_DIR} ]; then
  # build release default
  sh build.sh --with-tests
  LATEST_BUILD_DIR=$(ls -td build_* | head -n 1)
else
  DIR_SUFFIX=${LATEST_BUILD_DIR#build_}
  BUILD_TYPE=${DIR_SUFFIX} sh build.sh --with-tests
fi

make -C ${LATEST_BUILD_DIR} test