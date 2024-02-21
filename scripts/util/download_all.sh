#!/bin/sh

PROJ=$1
LOCAL_DIR="/home/mlandis/projects/phyddle/workspace/${PROJ}"
REMOTE_DIR="${llmf_wd}:/home/mlandis/projects/phyddle/workspace"

mkdir -p ${LOCAL_DIR}/train
scp -R ${REMOTE_DIR}/train ${LOCAL_DIR}/train

mkdir -p ${LOCAL_DIR}/estimate
scp -R ${REMOTE_DIR}/estimate ${LOCAL_DIR}/estimate

mkdir -p ${LOCAL_DIR}/plot
scp -R ${REMOTE_DIR}/plot ${LOCAL_DIR}/plot

