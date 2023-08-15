ROOT=`dirname "$0"`
ROOT=`cd "$ROOT"; pwd`
MACHINE_TYPE=$(uname -m)

export TENANN_HOME=${ROOT}

# Clean and prepare output dir
TENANN_OUTPUT=${TENANN_HOME}/output/
mkdir -p ${TENANN_OUTPUT}

# just for test
cd tenann
g++ -g -c version.cc -o libtenann.o -I..
ar rcs libtenann.a libtenann.o
cd -
cp tenann/*.h ${TENANN_OUTPUT}
cp tenann/libtenann.a ${TENANN_OUTPUT}
