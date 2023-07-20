#!/bin/sh

MODEL=$1
REMOTE="${llserver}"
LOCAL_DIR="../workspace/network/${MODEL}"
REMOTE_DIR="${REMOTE}:/home/mlandis/projects/phyddle/workspace/network"
#./clean_project.sh ${MODEL}
mkdir -p ${LOCAL_DIR}
scp "${REMOTE_DIR}/${MODEL}/sim_batchsize*_numepoch*nt*.*" ${LOCAL_DIR}

