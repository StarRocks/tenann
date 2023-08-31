# Copyright (c) Tencent, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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