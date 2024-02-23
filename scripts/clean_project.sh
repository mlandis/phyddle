#!/bin/sh
PROJECT_NAME=$1
WORK_DIR="./workspace/${PROJECT_NAME}"

# only clean project if string not empty
if [ -n "${PROJECT_NAME}" ]; then
    if [ -d "${WORK_DIR}/simulate" ]; then 
        rm -rf "${WORK_DIR}/simulate"
    fi
    if [ -d "${WORK_DIR}/format" ]; then 
        rm -rf "${WORK_DIR}/format"
    fi
    if [ -d "${WORK_DIR}/train" ]; then 
        rm -rf "${WORK_DIR}/train"
    fi
    if [ -d "${WORK_DIR}/estimate" ]; then 
        rm -rf "${WORK_DIR}/estimate"
    fi
    if [ -d "${WORK_DIR}/plot" ]; then 
        rm -rf "${WORK_DIR}/plot"
    fi
    if [ -d "${WORK_DIR}/log" ]; then 
        rm -rf "${WORK_DIR}/log"
    fi
fi
