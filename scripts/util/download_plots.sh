#!/bin/sh

MODEL=$1
REMOTE=$llserver
LOCAL_DIR="../workspace/plot/${MODEL}"
REMOTE_DIR="${REMOTE}:/home/mlandis/projects/phyddle/workspace/plot"
#./clean_project.sh ${MODEL}
mkdir -p ${LOCAL_DIR}
scp "${REMOTE_DIR}/${MODEL}/*.pdf" ${LOCAL_DIR}

