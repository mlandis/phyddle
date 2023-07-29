#!/bin/sh

MODEL=$1
REMOTE="${llserver}"
REMOTE_DIR="${REMOTE}:/home/mlandis/projects/phyddle/workspace/tensor_data"
LOCAL_DIR="../workspace/tensor_data/${MODEL}"
#./clean_project.sh ${MODEL}
mkdir -p ${LOCAL_DIR}
scp "${REMOTE_DIR}/${MODEL}/*.hdf5" ${LOCAL_DIR}
scp "${REMOTE_DIR}/${MODEL}/*.csv" ${LOCAL_DIR}


