#!/bin/sh
PROJECT_NAME=$1

# only clean project if string not empty
if [ -n "${PROJECT_NAME}" ]; then
    rm -rf ../workspace/simulate/${PROJECT_NAME}
    rm -rf ../workspace/format/${PROJECT_NAME}
    rm -rf ../workspace/train/${PROJECT_NAME}
    rm -rf ../workspace/estimate/${PROJECT_NAME}
    rm -rf ../workspace/plot/${PROJECT_NAME}
fi
