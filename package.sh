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
OPENBLAS_LIB=$(find ${TENANN_THIRDPARTY}/installed/lib -name "libopenblas*r*.a" | head -n 1)
if [ -z "$OPENBLAS_LIB" ]; then
    echo "Error: OpenBLAS library not found in ${TENANN_THIRDPARTY}/installed/lib"
    exit 1
fi
OPENBLAS_BASENAME=$(basename "$OPENBLAS_LIB")
echo "Detected OpenBLAS library: $OPENBLAS_BASENAME"

# Copy third-party libraries to the output directory
cp ${OPENBLAS_LIB} ${TENANN_OUTPUT}/tmp
cp ${TENANN_THIRDPARTY}/installed/lib/libfaiss.a ${TENANN_OUTPUT}/tmp

# Copy architecture-specific FAISS libraries
if [ -f "${TENANN_THIRDPARTY}/installed/lib/libfaiss_avx2.a" ]; then
    cp ${TENANN_THIRDPARTY}/installed/lib/libfaiss_avx2.a ${TENANN_OUTPUT}/tmp
fi
if [ -f "${TENANN_THIRDPARTY}/installed/lib/libfaiss_sve.a" ]; then
    cp ${TENANN_THIRDPARTY}/installed/lib/libfaiss_sve.a ${TENANN_OUTPUT}/tmp
fi

# Copy TenANN libraries
cp ${TENANN_OUTPUT}/lib/libtenann.a ${TENANN_OUTPUT}/tmp
if [ -f "${TENANN_OUTPUT}/lib/libtenann_avx2.a" ]; then
    cp ${TENANN_OUTPUT}/lib/libtenann_avx2.a ${TENANN_OUTPUT}/tmp
fi
if [ -f "${TENANN_OUTPUT}/lib/libtenann_sve.a" ]; then
    cp ${TENANN_OUTPUT}/lib/libtenann_sve.a ${TENANN_OUTPUT}/tmp
fi

# Merge all static libraries into one
cd ${TENANN_OUTPUT}/tmp

# For ARM64, use SVE variant as the default bundle; otherwise use base variant
if [[ "$MACHINE_TYPE" == "aarch64" ]] || [[ "$MACHINE_TYPE" == "arm64" ]]; then
    # On ARM64, create bundle from SVE libraries (if available)
    if [ -f "${TENANN_OUTPUT}/tmp/libtenann_sve.a" ]; then
        cat >libtenann-bundle.mri <<EOF
create libtenann-bundle.a
addlib libtenann_sve.a
addlib libfaiss_sve.a
addlib ${OPENBLAS_BASENAME}
save
end
EOF
        ar -M <libtenann-bundle.mri
        cp ${TENANN_OUTPUT}/tmp/libtenann-bundle.a ${TENANN_OUTPUT}/lib
        echo "Created libtenann-bundle.a (ARM SVE variant)"
    else
        # Fallback to base libraries if SVE not available
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
    fi
else
    # On x86_64, create base bundle
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

    # Create AVX2 variant if available
    if [ -f "${TENANN_OUTPUT}/tmp/libtenann_avx2.a" ]; then
        cat >libtenann-bundle-avx2.mri <<EOF
create libtenann-bundle-avx2.a
addlib libtenann_avx2.a
addlib libfaiss_avx2.a
addlib ${OPENBLAS_BASENAME}
save
end
EOF
        ar -M <libtenann-bundle-avx2.mri
        cp ${TENANN_OUTPUT}/tmp/libtenann-bundle-avx2.a ${TENANN_OUTPUT}/lib
        echo "Created libtenann-bundle-avx2.a"
    fi
fi

# Clean temporary directory
rm -rf ${TENANN_OUTPUT}/tmp

# Create final distribution package
RELEASE_VERSION="tenann-v0.5.1-BETA"
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

    # For x86_64, include both standard and AVX2 versions
    if [ -f "${TENANN_OUTPUT}/lib/libtenann-bundle.a" ]; then
        cp ${TENANN_OUTPUT}/lib/libtenann-bundle.a ${RELEASE_DIR}/lib/
        echo "  Added libtenann-bundle.a"
    else
        echo "Error: libtenann-bundle.a not found"
        exit 1
    fi

    if [ -f "${TENANN_OUTPUT}/lib/libtenann-bundle-avx2.a" ]; then
        cp ${TENANN_OUTPUT}/lib/libtenann-bundle-avx2.a ${RELEASE_DIR}/lib/
        echo "  Added libtenann-bundle-avx2.a"
    else
        echo "Warning: libtenann-bundle-avx2.a not found, skipping"
    fi

    PACKAGE_NAME="${RELEASE_VERSION}-x86_64.tar.gz"

elif [ "$MACHINE_TYPE" == "aarch64" ] || [ "$MACHINE_TYPE" == "arm64" ]; then
    echo "Detected ARM64 architecture"

    # For ARM64, libtenann-bundle.a is already the SVE variant
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
