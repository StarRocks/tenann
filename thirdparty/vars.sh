
#!/bin/bash
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

############################################################
# You may have to set variables bellow,
# which are used for compiling thirdparties and tenann itself.
############################################################

# --job param for *make*
# support macos
if [[ $(uname) == "Darwin" ]]; then
    default_parallel=$[$(sysctl -n hw.physicalcpu)/4+1]
else
    default_parallel=$[$(nproc)/4+1]
fi

# use the value if $PARALEL is already set, otherwise use $default_parallel
PARALLEL=${PARALLEL:-$default_parallel}

###################################################
# DO NOT change variables bellow unless you known
# what you are doing.
###################################################

# thirdparties will be downloaded and unpacked here
export TP_SOURCE_DIR=$TP_DIR/src

# thirdparties will be installed to here
export TP_INSTALL_DIR=$TP_DIR/installed

# patches for all thirdparties
export TP_PATCH_DIR=$TP_DIR/patches

# header files of all thirdparties will be intalled to here
export TP_INCLUDE_DIR=$TP_INSTALL_DIR/include

# libraries of all thirdparties will be intalled to here
export TP_LIB_DIR=$TP_INSTALL_DIR/lib

# all java libraries will be unpacked to here
export TP_JAR_DIR=$TP_INSTALL_DIR/lib/jar

#####################################################
# Download url, filename and unpacked filename
# of all thirdparties
#####################################################

# Definitions for architecture-related thirdparty
MACHINE_TYPE=$(uname -m)
# handle mac m1 platform, change arm64 to aarch64
if [[ "${MACHINE_TYPE}" == "arm64" ]]; then 
    MACHINE_TYPE="aarch64"
fi

VARS_TARGET=vars-${MACHINE_TYPE}.sh

if [ ! -f ${TP_DIR}/${VARS_TARGET} ]; then
    echo "${TP_DIR}/${VARS_TARGET} is missing".
    exit 1
fi
. ${TP_DIR}/${VARS_TARGET}

if [ -f /etc/lsb-release ]; then
    source /etc/lsb-release
    if [[ $DISTRIB_ID = "Ubuntu" && $DISTRIB_RELEASE =~ 22.* && -f ${TP_DIR}/vars-ubuntu22-${MACHINE_TYPE}.sh ]]; then
        . ${TP_DIR}/vars-ubuntu22-${MACHINE_TYPE}.sh
    fi
fi

# faiss
FAISS_DOWNLOAD="https://github.com/facebookresearch/faiss/archive/refs/tags/v1.7.3.tar.gz"
FAISS_NAME=faiss-v1.7.3.tar.gz
FAISS_SOURCE=faiss-1.7.3
FAISS_MD5SUM="632c5f465e80ebf10a7e2a54e5c853f7"

# nlohmann
NLOHMANN_JSON_DOWNLOAD="https://github.com/nlohmann/json/archive/refs/tags/v3.11.2.tar.gz"
NLOHMANN_JSON_NAME=json-3.11.2.tar.gz
NLOHMANN_JSON_SOURCE=json-3.11.2
NLOHMANN_JSON_MD5SUM="e8d56bc54621037842ee9f0aeae27746"

# all thirdparties which need to be downloaded is set in array TP_ARCHIVES
TP_ARCHIVES="FAISS NLOHMANN_JSON"
