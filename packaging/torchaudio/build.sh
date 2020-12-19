#!/usr/bin/env bash
set -ex

# export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
# export LD_LIBRARY_PATH="$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
# export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH
# export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
# export CFLAGS="-I$CUDA_HOME/include $CFLAGS"

BUILD_SOX=1 python setup.py install --single-version-externally-managed --record=record.txt
