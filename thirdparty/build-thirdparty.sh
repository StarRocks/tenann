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
bash ${TP_DIR}/download-thirdparty.sh

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

build_fmt() {
    check_if_source_exist $FMT_SOURCE
    cd $TP_SOURCE_DIR/$FMT_SOURCE
    mkdir -p build
    cd build
    rm -rf CMakeCache.txt CMakeFiles/
    $CMAKE_CMD -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${TP_INSTALL_DIR} ../ \
        -DCMAKE_INSTALL_LIBDIR=lib64 -G "${CMAKE_GENERATOR}" -DFMT_TEST=OFF
    ${BUILD_SYSTEM} -j$PARALLEL
    ${BUILD_SYSTEM} install
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
        -DCMAKE_Fortran_FLAGS:STRING="-fimplicit-none -frecursive" \
        -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON \
        ..

    $CMAKE_CMD --build . -j --target install
    rm -rf ${TP_INSTALL_DIR}/lib/cmake/lapack
    mv ${TP_INSTALL_DIR}/lib/cmake/$LAPACK_SOURCE ${TP_INSTALL_DIR}/lib/cmake/lapack
    cp -f ${TP_INSTALL_DIR}/lib/cmake/lapack/lapack-config.cmake ${TP_INSTALL_DIR}/lib/cmake/lapack/blas-config.cmake
}

# build_openblas() {
#     check_if_source_exist $OPENBLAS_SOURCE
#     cd $TP_SOURCE_DIR/$OPENBLAS_SOURCE
#     mkdir -p $BUILD_DIR
#     cd $BUILD_DIR
#     rm -rf CMakeCache.txt CMakeFiles/
#     $CMAKE_CMD -DCMAKE_INSTALL_PREFIX=${TP_INSTALL_DIR} \
#         -DCMAKE_INSTALL_LIBDIR=lib \
#         -DCMAKE_INSTALL_INCLUDEDIR=${TP_INSTALL_DIR}/include \
#         -DCMAKE_INSTALL_DATAROOTDIR=${TP_INSTALL_DIR}/lib/cmake \
#         -DDYNAMIC_ARCH=1 \
#         -DNO_SHARED=1 \
#         -DNO_AVX512=1 \
#         -DUSE_THREAD=0 \
#         -DUSE_OPENMP=0 \
#         ..
#     $CMAKE_CMD --build . -j$PARALLEL --target install
# }

build_openblas() {
    restore_compile_flags
    check_if_source_exist $OPENBLAS_SOURCE
    cd $TP_SOURCE_DIR/$OPENBLAS_SOURCE
    make clean
    # DYNAMIC_ARCH selects a kernel at run time, the way faiss does; DYNAMIC_LIST keeps only
    # the ISAs worth carrying, and TARGET names the oldest CPU the code built once -- LAPACK
    # above all -- may assume. DYNAMIC_LIST does not govern that shared body: left to itself
    # getarch picks the build machine's own core, which puts instructions no run-time check
    # guards into it. An array because DYNAMIC_LIST's value has a space in it.
    if [[ "${MACHINE_TYPE}" == "x86_64" ]]; then
        BLAS_FLAGS=(DYNAMIC_ARCH=1 TARGET=PRESCOTT "DYNAMIC_LIST=HASWELL SKYLAKEX" NO_SHARED=1 USE_THREAD=0 USE_OPENMP=0 USE_LOCKING=1 NOFORTRAN=1)
    elif [[ "${MACHINE_TYPE}" == "aarch64" ]]; then
        # Neoverse N1 (Graviton2, Ampere Altra), N2 (Yitian 710), V1 (Graviton3) and TSV110
        # (Kunpeng 920), out of seventeen. The list spans non-SVE and SVE cores on purpose:
        # with ARMV8 as the baseline, one package serves both.
        BLAS_FLAGS=(DYNAMIC_ARCH=1 TARGET=ARMV8 "DYNAMIC_LIST=NEOVERSEN1 NEOVERSEN2 NEOVERSEV1 TSV110" NO_SHARED=1 USE_THREAD=0 USE_OPENMP=0 USE_LOCKING=1 NO_SME=1 NOFORTRAN=1)
    else
        BLAS_FLAGS=(NO_SHARED=1 USE_THREAD=0 USE_OPENMP=0 USE_LOCKING=1)
    fi
    make -j$PARALLEL "${BLAS_FLAGS[@]}" libs netlib
    make PREFIX=${TP_INSTALL_DIR} "${BLAS_FLAGS[@]}" install
}
#faiss
build_faiss() {
    check_if_source_exist $FAISS_SOURCE
    cd $TP_SOURCE_DIR/$FAISS_SOURCE

    mkdir -p $BUILD_DIR
    cd $BUILD_DIR
    rm -rf CMakeCache.txt CMakeFiles/
    echo "machine type:" $MACHINE_TYPE

    # Not "dd". FAISS's dynamic-dispatch mode compiles every per-ISA kernel into the one
    # faiss target, and templates that are NOT parameterised by SIMD level -- the comment on
    # PQCodeDistanceScalar says so itself -- are then emitted from avx2.cpp and avx512.cpp
    # under the same mangled name with different code. They are weak symbols, so the linker
    # keeps whichever it meets first: pick the AVX-512 one and the baseline and AVX2 paths
    # execute AVX-512 too, which SIGILLs on any CPU without it. It survives testing on an
    # AVX-512 machine and dies on the first that lacks it.
    #
    # A fixed opt level keeps each library to one ISA, so nothing can collide across levels.
    # The aarch64 SVE package is a second, separate build of the whole thing:
    # FAISS_OPT_LEVEL=sve is the only level at which faiss_sve is built and installed
    # (every other level marks it EXCLUDE_FROM_ALL). Pair it with build.sh --with-sve.
    #   non-SVE arm64:  ./build-thirdparty.sh                      && ./build.sh
    #   SVE arm64:      FAISS_OPT_LEVEL_OVERRIDE=sve ./build-thirdparty.sh && ./build.sh --with-sve
    if [ -n "${FAISS_OPT_LEVEL_OVERRIDE:-}" ]; then
        FAISS_OPT_LEVEL=${FAISS_OPT_LEVEL_OVERRIDE}
    elif [[ "${MACHINE_TYPE}" == "x86_64" ]]; then
        FAISS_OPT_LEVEL=avx2
    else
        FAISS_OPT_LEVEL=generic
    fi
    echo "FAISS_OPT_LEVEL: $FAISS_OPT_LEVEL"

    $CMAKE_CMD -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=${TP_INSTALL_DIR} \
        -DCMAKE_INSTALL_DATAROOTDIR=${TP_INSTALL_DIR}/lib/cmake \
        -DCMAKE_INSTALL_LIBDIR=lib \
        -DCMAKE_CXX_COMPILER=$TENANN_GCC_HOME/bin/g++ \
        -DCMAKE_C_COMPILER=$TENANN_GCC_HOME/bin/gcc \
        -DFAISS_ENABLE_GPU=OFF \
        -DFAISS_ENABLE_PYTHON=OFF \
        -DFAISS_OPT_LEVEL=${FAISS_OPT_LEVEL} \
        -DBUILD_SHARED_LIBS=OFF \
        -DBUILD_TESTING=OFF \
        -DFAISS_ENABLE_MKL=OFF \
        ..

    # The faiss patch makes OpenMP optional, so a failed detection would
    # silently produce a single-threaded faiss. Fail loudly instead.
    if ! grep -q "^OpenMP_CXX_FLAGS:STRING=.*-fopenmp" CMakeCache.txt; then
        echo "ERROR: faiss was configured without OpenMP (-fopenmp missing)."
        echo "       Check FindOpenMP detection for ${TENANN_GCC_HOME}/bin/g++."
        exit 1
    fi

    ${BUILD_SYSTEM} -j$PARALLEL
    ${BUILD_SYSTEM} install

    cp -f ${TP_INSTALL_DIR}/lib/cmake/faiss/faiss-config.cmake ${TP_INSTALL_DIR}/lib/cmake/faiss/faiss_${FAISS_OPT_LEVEL}-config.cmake
}


# gtest
build_gtest() {
    check_if_source_exist $GTEST_SOURCE

    cd $TP_SOURCE_DIR/$GTEST_SOURCE
    mkdir -p $BUILD_DIR
    cd $BUILD_DIR
    rm -rf CMakeCache.txt CMakeFiles/
    $CMAKE_CMD -G "${CMAKE_GENERATOR}" -DCMAKE_INSTALL_PREFIX=$TP_INSTALL_DIR -DCMAKE_INSTALL_LIBDIR=lib \
        -DCMAKE_POSITION_INDEPENDENT_CODE=On ../
    ${BUILD_SYSTEM} -j$PARALLEL
    ${BUILD_SYSTEM} install
}

# pybind11
build_pybind11() {
    check_if_source_exist $PYBIND11_SOURCE

    cd $TP_SOURCE_DIR/$PYBIND11_SOURCE
    mkdir -p $BUILD_DIR
    cd $BUILD_DIR
    rm -rf CMakeCache.txt CMakeFiles/
    $CMAKE_CMD -G "${CMAKE_GENERATOR}" -DCMAKE_INSTALL_PREFIX=$TP_INSTALL_DIR -DCMAKE_INSTALL_LIBDIR=lib \
        -DCMAKE_POSITION_INDEPENDENT_CODE=On ../
    ${BUILD_SYSTEM} -j$PARALLEL
    ${BUILD_SYSTEM} install
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
export GLOBAL_CFLAGS="-fPIC -static-libstdc++ -static-libgcc -O3 -fno-omit-frame-pointer -std=gnu99 -fPIC -g -D_POSIX_C_SOURCE=199309L"
export GLOBAL_CXXFLAGS="-fPIC -static-libstdc++ -static-libgcc -O3 -fno-omit-frame-pointer -Wno-class-memaccess -fPIC -g"

# set those GLOBAL_*FLAGS to the CFLAGS/CXXFLAGS/CPPFLAGS
export CPPFLAGS=$GLOBAL_CPPFLAGS
export CXXFLAGS=$GLOBAL_CXXFLAGS
export CFLAGS=$GLOBAL_CFLAGS

build_fmt
build_openblas # must before faiss
build_faiss
build_gtest
# build_pybind11

# strip unnecessary debug symbol for binaries in thirdparty
strip_binary

echo "Finished to build all thirdparties"
