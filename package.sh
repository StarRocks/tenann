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

# Detect OpenBLAS library version dynamically
# Resolve through libopenblas.a, the canonical name OpenBLAS installs. Globbing for
# libopenblas*r*.a and taking the first hit picks by directory order, so a stale
# archive left by an earlier build with different flags wins silently: a leftover
# libopenblas_haswell-r*.a is enough to ship a bundle with AVX2 baked in and no
# runtime dispatch, which is exactly what this build is meant to stop doing.
OPENBLAS_LINK=${TENANN_THIRDPARTY}/installed/lib/libopenblas.a
if [ ! -e "$OPENBLAS_LINK" ]; then
    echo "Error: ${OPENBLAS_LINK} not found. Build OpenBLAS first."
    exit 1
fi
OPENBLAS_LIB=$(readlink -f "$OPENBLAS_LINK")
OPENBLAS_BASENAME=$(basename "$OPENBLAS_LIB")
echo "Detected OpenBLAS library: $OPENBLAS_BASENAME"

# Copy third-party libraries to the output directory
cp ${OPENBLAS_LIB} ${TENANN_OUTPUT}/tmp
cp ${TENANN_THIRDPARTY}/installed/lib/libfaiss.a ${TENANN_OUTPUT}/tmp

# Copy the TenANN library
cp ${TENANN_OUTPUT}/lib/libtenann.a ${TENANN_OUTPUT}/tmp

# Merge all static libraries into one
cd ${TENANN_OUTPUT}/tmp

# One bundle serves every CPU now: faiss selects its kernels at runtime, so there
# is nothing left to specialize per ISA.
cat >libtenann-bundle.mri <<EOF
create libtenann-bundle.a
addlib libtenann.a
addlib libfaiss.a
addlib ${OPENBLAS_BASENAME}
save
end
EOF
ar -M <libtenann-bundle.mri
cp ${TENANN_OUTPUT}/tmp/libtenann-bundle.a ${TENANN_OUTPUT}/lib
echo "Created libtenann-bundle.a"

# Clean temporary directory
rm -rf ${TENANN_OUTPUT}/tmp

# Create final distribution package
RELEASE_VERSION="tenann-v0.5.0-RELEASE"
RELEASE_DIR="${TENANN_HOME}/${RELEASE_VERSION}"

echo "Creating release package: ${RELEASE_VERSION}"

# Clean up any previous release directory
rm -rf ${RELEASE_DIR}
mkdir -p ${RELEASE_DIR}/lib

# Copy include directory
echo "Copying headers from ${TENANN_OUTPUT}/include to ${RELEASE_DIR}/include"
cp -r ${TENANN_OUTPUT}/include ${RELEASE_DIR}/

# Copy bundled libraries based on architecture
if [ "$MACHINE_TYPE" == "x86_64" ]; then
    echo "Detected x86_64 architecture"

    if [ -f "${TENANN_OUTPUT}/lib/libtenann-bundle.a" ]; then
        cp ${TENANN_OUTPUT}/lib/libtenann-bundle.a ${RELEASE_DIR}/lib/
        echo "  Added libtenann-bundle.a"
    else
        echo "Error: libtenann-bundle.a not found"
        exit 1
    fi

    PACKAGE_NAME="${RELEASE_VERSION}-x86_64.tar.gz"

elif [ "$MACHINE_TYPE" == "aarch64" ] || [ "$MACHINE_TYPE" == "arm64" ]; then
    echo "Detected ARM64 architecture"

    if [ -f "${TENANN_OUTPUT}/lib/libtenann-bundle.a" ]; then
        cp ${TENANN_OUTPUT}/lib/libtenann-bundle.a ${RELEASE_DIR}/lib/
        echo "  Added libtenann-bundle.a"
    else
        echo "Error: libtenann-bundle.a not found for ARM64"
        exit 1
    fi

    PACKAGE_NAME="${RELEASE_VERSION}-arm64.tar.gz"

else
    echo "Warning: Unknown architecture $MACHINE_TYPE, using generic package name"

    # For other architectures, just copy standard bundle
    if [ -f "${TENANN_OUTPUT}/lib/libtenann-bundle.a" ]; then
        cp ${TENANN_OUTPUT}/lib/libtenann-bundle.a ${RELEASE_DIR}/lib/
        echo "  Added libtenann-bundle.a"
    else
        echo "Error: libtenann-bundle.a not found"
        exit 1
    fi

    PACKAGE_NAME="${RELEASE_VERSION}-${MACHINE_TYPE}.tar.gz"
fi

# Create tarball
cd ${TENANN_HOME}
tar czf ${PACKAGE_NAME} ${RELEASE_VERSION}

echo ""
echo "========================================="
echo "Release package created successfully!"
echo "Package: ${TENANN_HOME}/${PACKAGE_NAME}"
echo "Contents:"
tar tzf ${PACKAGE_NAME}
echo "========================================="
