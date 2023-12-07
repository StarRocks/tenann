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

ROOT=$(dirname "$0")
ROOT=$(
    cd "$ROOT"
    pwd
)
MACHINE_TYPE=$(uname -m)

export TENANN_HOME=${ROOT}

if [ -z $BUILD_TYPE ]; then
    export BUILD_TYPE=Release
fi

set -eo pipefail
. ${TENANN_HOME}/env.sh

TENANN_OUTPUT=${TENANN_HOME}/output
rm -rf ${TENANN_OUTPUT}/tmp
mkdir -p ${TENANN_OUTPUT}/tmp

# Copy all third-party libraries to the output directory
cp /opt/gcc/usr/lib64/libquadmath.a ${TENANN_OUTPUT}/tmp
cp /usr/local/lib/libgfortran.a ${TENANN_OUTPUT}/tmp
cp /opt/gcc/usr/lib64/libgomp.a ${TENANN_OUTPUT}/tmp
cp ${TENANN_THIRDPARTY}/installed/lib/libopenblas-r0.3.25.a ${TENANN_OUTPUT}/tmp
cp ${TENANN_THIRDPARTY}/installed/lib/libfaiss.a ${TENANN_OUTPUT}/tmp
cp ${TENANN_THIRDPARTY}/installed/lib/libfaiss_avx2.a ${TENANN_OUTPUT}/tmp
cp ${TENANN_OUTPUT}/lib/libtenann.a ${TENANN_OUTPUT}/tmp
cp ${TENANN_OUTPUT}/lib/libtenann_avx2.a ${TENANN_OUTPUT}/tmp

# Merge all static libraries into one
cd ${TENANN_OUTPUT}/tmp
echo "create libtenann-bundle.a
addlib libtenann.a
addlib libfaiss.a
addlib libopenblas-r0.3.25.a
addlib libgomp.a
addlib libgfortran.a
addlib libquadmath.a
save
end" >libtenann-bundle.mri

ar -M <libtenann-bundle.mri
cp ${TENANN_OUTPUT}/tmp/libtenann-bundle.a ${TENANN_OUTPUT}/lib

# Merge all static libraries into one
cd ${TENANN_OUTPUT}/tmp
echo "create libtenann-bundle-avx2.a
addlib libtenann_avx2.a
addlib libfaiss_avx2.a
addlib libopenblas-r0.3.25.a
addlib libgomp.a
addlib libgfortran.a
addlib libquadmath.a
save
end" >libtenann-bundle-avx2.mri

ar -M <libtenann-bundle-avx2.mri
cp ${TENANN_OUTPUT}/tmp/libtenann-bundle-avx2.a ${TENANN_OUTPUT}/lib

# Clean temporary directory
rm -rf ${TENANN_OUTPUT}/tmp