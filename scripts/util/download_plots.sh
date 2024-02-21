#!/bin/sh

PROJ=$1
REMOTE=$llbh
LOCAL_DIR="./workspace/${PROJ}/plot"
REMOTE_DIR="${REMOTE}:/home/mlandis/projects/phyddle/workspace"
mkdir -p ${LOCAL_DIR}
scp "${REMOTE_DIR}/${PROJ}/plot/*.pdf" ${LOCAL_DIR}

