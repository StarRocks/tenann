ROOT=`dirname "$0"`
ROOT=`cd "$ROOT"; pwd`
MACHINE_TYPE=$(uname -m)

export TENANN_HOME=${ROOT}

# Clean and prepare output dir
TENANN_OUTPUT=${TENANN_HOME}/output/
mkdir -p ${TENANN_OUTPUT}

cmake -B build
make -C build -j
cp -r ${TENANN_HOME}/build64_release ${TENANN_OUTPUT}
# just for test
# cd ${TENANN_HOME}/tenann
# g++ -g -c version.cc -o libtenann.o -I..
# ar rcs libtenann.a libtenann.o
# cd -
# cp ${TENANN_HOME}/tenann/*.h ${TENANN_OUTPUT}
# cp ${TENANN_HOME}/tenann/libtenann.a ${TENANN_OUTPUT}
