#!/bin/sh
PROJECT_NAME=$1

# only clean project if string not empty
if [ -n "${PROJECT_NAME}" ]; then
    rm -rf ../workspace/raw_data/${PROJECT_NAME}
    rm -rf ../workspace/tensor_data/${PROJECT_NAME}
    rm -rf ../workspace/network/${PROJECT_NAME}
    rm -rf ../workspace/plot/${PROJECT_NAME}
fi
