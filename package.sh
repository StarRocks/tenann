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

# Function to find library in multiple possible locations
find_library() {
    local lib_name=$1
    local search_paths=(
        "/opt/gcc/usr/lib64"
        "/usr/local/lib"
        "/usr/lib"
        "/usr/lib64"
        "/usr/lib/gcc/*/*"
        "${TENANN_GCC_HOME}/lib64"
        "${TENANN_GCC_HOME}/lib"
    )

    for path in "${search_paths[@]}"; do
        local found=$(find $path -name "$lib_name" 2>/dev/null | head -n 1)
        if [ -n "$found" ]; then
            echo "$found"
            return 0
        fi
    done

    return 1
}

# Detect OpenBLAS library version dynamically
OPENBLAS_LIB=$(find ${TENANN_THIRDPARTY}/installed/lib -name "libopenblas-r*.a" | head -n 1)
if [ -z "$OPENBLAS_LIB" ]; then
    echo "Error: OpenBLAS library not found in ${TENANN_THIRDPARTY}/installed/lib"
    exit 1
fi
OPENBLAS_BASENAME=$(basename "$OPENBLAS_LIB")
echo "Detected OpenBLAS library: $OPENBLAS_BASENAME"

# Find optional GCC libraries (may not be available on all platforms, e.g. ARM64)
LIBQUADMATH=$(find_library "libquadmath.a" || echo "")
LIBGFORTRAN=$(find_library "libgfortran.a" || echo "")
LIBGOMP=$(find_library "libgomp.a" || echo "")

echo "Detected libraries:"
if [ -n "$LIBQUADMATH" ]; then
    echo "  libquadmath: $LIBQUADMATH"
else
    echo "  libquadmath: not found (optional, skipping)"
fi
if [ -n "$LIBGFORTRAN" ]; then
    echo "  libgfortran: $LIBGFORTRAN"
else
    echo "  libgfortran: not found (optional, skipping)"
fi
if [ -n "$LIBGOMP" ]; then
    echo "  libgomp: $LIBGOMP"
else
    echo "  libgomp: not found (optional, skipping)"
fi

# Copy all third-party libraries to the output directory
if [ -n "$LIBQUADMATH" ]; then
    cp "$LIBQUADMATH" ${TENANN_OUTPUT}/tmp
fi
if [ -n "$LIBGFORTRAN" ]; then
    cp "$LIBGFORTRAN" ${TENANN_OUTPUT}/tmp
fi
if [ -n "$LIBGOMP" ]; then
    cp "$LIBGOMP" ${TENANN_OUTPUT}/tmp
fi
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
cat >libtenann-bundle.mri <<EOF
create libtenann-bundle.a
addlib libtenann.a
addlib libfaiss.a
addlib ${OPENBLAS_BASENAME}
EOF

if [ -n "$LIBGOMP" ]; then
    echo "addlib libgomp.a" >>libtenann-bundle.mri
fi
if [ -n "$LIBGFORTRAN" ]; then
    echo "addlib libgfortran.a" >>libtenann-bundle.mri
fi
if [ -n "$LIBQUADMATH" ]; then
    echo "addlib libquadmath.a" >>libtenann-bundle.mri
fi

cat >>libtenann-bundle.mri <<EOF
save
end
EOF

ar -M <libtenann-bundle.mri
cp ${TENANN_OUTPUT}/tmp/libtenann-bundle.a ${TENANN_OUTPUT}/lib
echo "Created libtenann-bundle.a"

# Merge all static libraries into one (AVX2 variant)
if [ -f "${TENANN_OUTPUT}/tmp/libtenann_avx2.a" ]; then
    cd ${TENANN_OUTPUT}/tmp
    cat >libtenann-bundle-avx2.mri <<EOF
create libtenann-bundle-avx2.a
addlib libtenann_avx2.a
addlib libfaiss_avx2.a
addlib ${OPENBLAS_BASENAME}
EOF

    if [ -n "$LIBGOMP" ]; then
        echo "addlib libgomp.a" >>libtenann-bundle-avx2.mri
    fi
    if [ -n "$LIBGFORTRAN" ]; then
        echo "addlib libgfortran.a" >>libtenann-bundle-avx2.mri
    fi
    if [ -n "$LIBQUADMATH" ]; then
        echo "addlib libquadmath.a" >>libtenann-bundle-avx2.mri
    fi

    cat >>libtenann-bundle-avx2.mri <<EOF
save
end
EOF

    ar -M <libtenann-bundle-avx2.mri
    cp ${TENANN_OUTPUT}/tmp/libtenann-bundle-avx2.a ${TENANN_OUTPUT}/lib
    echo "Created libtenann-bundle-avx2.a"
fi

# Merge all static libraries into one (SVE variant for ARM64)
if [ -f "${TENANN_OUTPUT}/tmp/libtenann_sve.a" ]; then
    cd ${TENANN_OUTPUT}/tmp
    cat >libtenann-bundle-sve.mri <<EOF
create libtenann-bundle-sve.a
addlib libtenann_sve.a
addlib libfaiss_sve.a
addlib ${OPENBLAS_BASENAME}
EOF

    if [ -n "$LIBGOMP" ]; then
        echo "addlib libgomp.a" >>libtenann-bundle-sve.mri
    fi
    if [ -n "$LIBGFORTRAN" ]; then
        echo "addlib libgfortran.a" >>libtenann-bundle-sve.mri
    fi
    if [ -n "$LIBQUADMATH" ]; then
        echo "addlib libquadmath.a" >>libtenann-bundle-sve.mri
    fi

    cat >>libtenann-bundle-sve.mri <<EOF
save
end
EOF

    ar -M <libtenann-bundle-sve.mri
    cp ${TENANN_OUTPUT}/tmp/libtenann-bundle-sve.a ${TENANN_OUTPUT}/lib
    echo "Created libtenann-bundle-sve.a"
fi

# Clean temporary directory
rm -rf ${TENANN_OUTPUT}/tmp
