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

cd $TENANN_HOME
if [ -z $TENANN_VERSION ]; then
    tag_name=$(git describe --tags --exact-match 2>/dev/null)
    branch_name=$(git symbolic-ref -q --short HEAD)
    if [ ! -z $tag_name ]; then
        export TENANN_VERSION=$tag_name
    elif [ ! -z $branch_name ]; then
        export TENANN_VERSION=$branch_name
    else
        export TENANN_VERSION=$(git rev-parse --short HEAD)
    fi
fi

if [ -z $TENANN_COMMIT_HASH]; then
    export TENANN_COMMIT_HASH=$(git rev-parse --short HEAD)
fi

set -eo pipefail
. ${TENANN_HOME}/env.sh

if [[ $OSTYPE == darwin* ]]; then
    PARALLEL=$(sysctl -n hw.ncpu)
    # We know for sure that build-thirdparty.sh will fail on darwin platform, so just skip the step.
else
    if [[ ! -f ${TENANN_THIRDPARTY}/installed/include/faiss/Index.h ]]; then
        echo "Thirdparty libraries need to be build ..."
        sh ${TENANN_THIRDPARTY}/build-thirdparty.sh
    fi
    PARALLEL=$(($(nproc) / 4 + 1))
fi

# Check args
usage() {
    echo "
Usage: $0 <options>
  Optional options:
     --clean            clean and build target
     --with-examples    build tenann with examples
     --with-tests       build tenann with tests
     -j                 build Backend parallel

  Eg.
    $0                               build tenann
    $0 --clean                       clean and build tenann
    $0 --with-examples --with-tests  build tenann with examples and tests
    BUILD_TYPE=build_type $0         build tenann in different mode (build_type could be Release, Debug, or Asan. Default value is Release. To build Backend in Debug mode, you can execute: BUILD_TYPE=Debug ./build.sh --tenann)
  "
    exit 1
}

OPTS=$(getopt \
    -n $0 \
    -o '' \
    -o 'h' \
    -l 'with-examples' \
    -l 'with-tests' \
    -l 'tenann' \
    -l 'clean' \
    -o 'j:' \
    -l 'help' \
    -- "$@")

if [ $? != 0 ]; then
    usage
fi

eval set -- "$OPTS"

BUILD_TENANN=
CLEAN=
WITH_EXAMPLES=
WITH_TESTS=
MSG=""
MSG_TENANN="libtenann.a"

HELP=0
if [ $# == 1 ]; then
    # default
    BUILD_TENANN=1
    CLEAN=0
    WITH_EXAMPLES=OFF
    WITH_TESTS=OFF
elif [[ $OPTS =~ "-j" ]] && [ $# == 3 ]; then
    # default
    BUILD_TENANN=1
    CLEAN=0
    WITH_EXAMPLES=OFF
    WITH_TESTS=OFF
    PARALLEL=$2
else
    BUILD_TENANN=1
    CLEAN=0
    WITH_EXAMPLES=OFF
    WITH_TESTS=OFF
    while true; do
        case "$1" in
        --tenann)
            BUILD_TENANN=1
            shift
            ;;
        --clean)
            CLEAN=1
            shift
            ;;
        --with-examples)
            WITH_EXAMPLES=ON
            shift
            ;;
        --with-tests)
            WITH_TESTS=ON
            shift
            ;;
        -h)
            HELP=1
            shift
            ;;
        --help)
            HELP=1
            shift
            ;;
        -j)
            PARALLEL=$2
            shift 2
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Internal error"
            exit 1
            ;;
        esac
    done
fi

if [[ ${HELP} -eq 1 ]]; then
    usage
    exit
fi

if [ ${CLEAN} -eq 1 -a ${BUILD_TENANN} -eq 0 ]; then
    echo "--clean can not be specified without --tenann"
    exit 1
fi

echo "Get params:
    TENANN_CMAKE_TYPE   -- $BUILD_TYPE
    BUILD_TENANN        -- $BUILD_TENANN
    CLEAN               -- $CLEAN
    WITH_EXAMPLES       -- $WITH_EXAMPLES
    WITH_TESTS          -- $WITH_TESTS
    PARALLEL            -- $PARALLEL
"
if [ ${BUILD_TENANN} -eq 1 ]; then
    # Clean and build tenann
    if ! ${CMAKE_CMD} --version; then
        echo "Error: cmake is not found"
        exit 1
    fi

    CMAKE_BUILD_TYPE=$BUILD_TYPE
    echo "Build tenann: ${CMAKE_BUILD_TYPE}"
    CMAKE_BUILD_DIR=${TENANN_HOME}/build_${CMAKE_BUILD_TYPE}

    if [ ${CLEAN} -eq 1 ]; then
        rm -rf ${CMAKE_BUILD_DIR}
        rm -rf ${TENANN_HOME}/output/
    fi
    mkdir -p ${CMAKE_BUILD_DIR}

    cd ${CMAKE_BUILD_DIR}
    rm -rf CMakeCache.txt CMakeFiles/

    ${CMAKE_CMD} -G "${CMAKE_GENERATOR}" \
        -DTENANN_THIRDPARTY=${TENANN_THIRDPARTY} \
        -DTENANN_HOME=${TENANN_HOME} \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DBUILD_TESTS=${WITH_TESTS} \
        -DBUILD_EXAMPLES=${WITH} \
        ..

    time ${BUILD_SYSTEM} -j${PARALLEL}
# ${BUILD_SYSTEM} install
fi

cd ${TENANN_HOME}

# Clean and prepare output dir
TENANN_OUTPUT=${TENANN_HOME}/output/
mkdir -p ${TENANN_OUTPUT}

if [ ${BUILD_TENANN} -eq 1 ]; then
    rm -rf ${TENANN_OUTPUT}/tenann

    install -d ${TENANN_OUTPUT}/tenann/lib \
        ${TENANN_OUTPUT}/tenann/include

    cp -r -p ${CMAKE_BUILD_DIR}/tenann/*.a ${TENANN_OUTPUT}/tenann/lib/
    cp -r -p ${CMAKE_BUILD_DIR}/tenann/include ${TENANN_OUTPUT}/tenann/

    MSG="${MSG} √ ${MSG_TENANN}"
fi

cp -r -p "${TENANN_HOME}/LICENSE.txt" "${TENANN_OUTPUT}/LICENSE.txt"

echo "***************************************"
echo "Successfully build TenANN ${MSG}"
echo "***************************************"

exit 0
