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

#################################################################################
# This script will
# 1. Check prerequisite libraries. Including:
#    cmake byacc flex automake libtool binutils-dev libiberty-dev bison
# 2. Compile and install all thirdparties which are downloaded
#    using *download-thirdparty.sh*.
#
# This script will run *download-thirdparty.sh* once again
# to check if all thirdparties have been downloaded, unpacked and patched.
#################################################################################
set -e

curdir=$(dirname "$0")
curdir=$(
    cd "$curdir"
    pwd
)

export TENANN_HOME=${TENANN_HOME:-$curdir/..}
export TP_DIR=$curdir

# include custom environment variables
if [[ -f ${TENANN_HOME}/env.sh ]]; then
    . ${TENANN_HOME}/env.sh
fi

if [[ ! -f ${TP_DIR}/download-thirdparty.sh ]]; then
    echo "Download thirdparty script is missing".
    exit 1
fi

if [ ! -f ${TP_DIR}/vars.sh ]; then
    echo "vars.sh is missing".
    exit 1
fi
. ${TP_DIR}/vars.sh

cd $TP_DIR

# Download thirdparties.
sh ${TP_DIR}/download-thirdparty.sh

# set COMPILER
if [[ ! -z ${TENANN_GCC_HOME} ]]; then
    export CC=${TENANN_GCC_HOME}/bin/gcc
    export CPP=${TENANN_GCC_HOME}/bin/cpp
    export CXX=${TENANN_GCC_HOME}/bin/g++
    export PATH=${TENANN_GCC_HOME}/bin:$PATH
else
    echo "TENANN_GCC_HOME environment variable is not set"
    exit 1
fi

# prepare installed prefix
mkdir -p ${TP_DIR}/installed

check_prerequest() {
    local CMD=$1
    local NAME=$2
    if ! $CMD; then
        echo $NAME is missing
        exit 1
    else
        echo $NAME is found
    fi
}

# sudo apt-get install cmake
# sudo yum install cmake
check_prerequest "${CMAKE_CMD} --version" "cmake"

# sudo apt-get install automake
# sudo yum install automake
check_prerequest "automake --version" "automake"

# sudo apt-get install libtool
# sudo yum install libtool
# check_prerequest "libtoolize --version" "libtool"

# sudo apt-get install ldconfig
# sudo yum install ldconfig
check_prerequest "ldconfig --version" "ldconfig"

BUILD_SYSTEM=${BUILD_SYSTEM:-make}

#########################
# build all thirdparties
#########################

# Name of cmake build directory in each thirdpary project.
# Do not use `build`, because many projects contained a file named `BUILD`
# and if the filesystem is not case sensitive, `mkdir` will fail.
BUILD_DIR=tenann_build
MACHINE_TYPE=$(uname -m)

# handle mac m1 platform, change arm64 to aarch64
if [[ "${MACHINE_TYPE}" == "arm64" ]]; then
    MACHINE_TYPE="aarch64"
fi

echo "machine type : $MACHINE_TYPE"

check_if_source_exist() {
    if [ -z $1 ]; then
        echo "dir should specified to check if exist."
        exit 1
    fi

    if [ ! -d $TP_SOURCE_DIR/$1 ]; then
        echo "$TP_SOURCE_DIR/$1 does not exist."
        exit 1
    fi
    echo "===== begin build $1"
}

check_if_archieve_exist() {
    if [ -z $1 ]; then
        echo "archieve should specified to check if exist."
        exit 1
    fi

    if [ ! -f $TP_SOURCE_DIR/$1 ]; then
        echo "$TP_SOURCE_DIR/$1 does not exist."
        exit 1
    fi
}

build_lapack() {
    check_if_source_exist $LAPACK_SOURCE
    cd $TP_SOURCE_DIR/$LAPACK_SOURCE
    mkdir -p $BUILD_DIR
    cd $BUILD_DIR
    rm -rf CMakeCache.txt CMakeFiles/
    $CMAKE_CMD -DCMAKE_INSTALL_PREFIX=${TP_INSTALL_DIR} \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DCMAKE_INSTALL_INCLUDEDIR=${TP_INSTALL_DIR}/include/lapack \
    -DCMAKE_INSTALL_DATAROOTDIR=${TP_INSTALL_DIR}/lib/cmake \
    -DLAPACKE:BOOL=OFF \
    -DCBLAS:BOOL=OFF \
    -DCMAKE_Fortran_FLAGS:STRING="-fimplicit-none -frecursive -fcheck=all" \
    ..

    $CMAKE_CMD --build . -j --target install
    rm -rf ${TP_INSTALL_DIR}/lib/cmake/lapack
    mv ${TP_INSTALL_DIR}/lib/cmake/$LAPACK_SOURCE ${TP_INSTALL_DIR}/lib/cmake/lapack
    cp -f ${TP_INSTALL_DIR}/lib/cmake/lapack/lapack-config.cmake ${TP_INSTALL_DIR}/lib/cmake/lapack/blas-config.cmake
}

#faiss
build_faiss() {
    check_if_source_exist $FAISS_SOURCE
    cd $TP_SOURCE_DIR/$FAISS_SOURCE

    # faiss 库依赖的lapack 动态库，导致链接了 faiss 的 tenann 也会依赖 lapack 动态库，幂等修改 faiss/CMakeLists.txt，链接静态 lapack 库
    sed -i 's/find_package(BLAS REQUIRED)/find_package(BLAS REQUIRED PATHS ${CMAKE_INSTALL_DATAROOTDIR}\/lapack)/g' faiss/CMakeLists.txt
    sed -i 's/find_package(LAPACK REQUIRED)/find_package(LAPACK REQUIRED PATHS ${CMAKE_INSTALL_DATAROOTDIR}\/lapack)/g' faiss/CMakeLists.txt

    mkdir -p $BUILD_DIR
    cd $BUILD_DIR
    rm -rf CMakeCache.txt CMakeFiles/
    $CMAKE_CMD -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=${TP_INSTALL_DIR} \
    -DCMAKE_INSTALL_DATAROOTDIR=${TP_INSTALL_DIR}/lib/cmake \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DCMAKE_CXX_COMPILER=$TENANN_GCC_HOME/bin/g++ \
    -DCMAKE_C_COMPILER=$TENANN_GCC_HOME/bin/gcc \
    -DFAISS_ENABLE_GPU=OFF \
    -DFAISS_ENABLE_PYTHON=OFF \
    -DFAISS_OPT_LEVEL=avx2 \
    -DBUILD_SHARED_LIBS=OFF \
    -DBUILD_TESTING=OFF \
    ..

    ${BUILD_SYSTEM} -j$PARALLEL
    ${BUILD_SYSTEM} install

    cp -f ${TP_INSTALL_DIR}/lib/cmake/faiss/faiss-config.cmake ${TP_INSTALL_DIR}/lib/cmake/faiss/faiss_avx2-config.cmake
}

# restore cxxflags/cppflags/cflags to default one
restore_compile_flags() {
    # c preprocessor flags
    export CPPFLAGS=$GLOBAL_CPPFLAGS
    # c flags
    export CFLAGS=$GLOBAL_CFLAGS
    # c++ flags
    export CXXFLAGS=$GLOBAL_CXXFLAGS
}

strip_binary() {
    # strip binary tools and ignore any errors
    echo "Strip binaries in $TP_INSTALL_DIR/bin/ ..."
    strip $TP_INSTALL_DIR/bin/* 2>/dev/null || true
}

# set GLOBAL_C*FLAGS for easy restore in each sub build process
export GLOBAL_CPPFLAGS="-I ${TP_INCLUDE_DIR}"
# https://stackoverflow.com/questions/42597685/storage-size-of-timespec-isnt-known
export GLOBAL_CFLAGS="-static-libstdc++ -static-libgcc -O3 -fno-omit-frame-pointer -std=c99 -fPIC -g -D_POSIX_C_SOURCE=199309L"
export GLOBAL_CXXFLAGS="-static-libstdc++ -static-libgcc -O3 -fno-omit-frame-pointer -Wno-class-memaccess -fPIC -g"

# set those GLOBAL_*FLAGS to the CFLAGS/CXXFLAGS/CPPFLAGS
export CPPFLAGS=$GLOBAL_CPPFLAGS
export CXXFLAGS=$GLOBAL_CXXFLAGS
export CFLAGS=$GLOBAL_CFLAGS

build_lapack # must before faiss
build_faiss

# strip unnecessary debug symbol for binaries in thirdparty
strip_binary

echo "Finished to build all thirdparties"
