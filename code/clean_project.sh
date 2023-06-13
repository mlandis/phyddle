#!/bin/sh
PROJECT_NAME=$1

# only clean project if string not empty
if [ -n "${PROJECT_NAME}" ]; then
    rm -rf ../raw_data/${PROJECT_NAME}
    rm -rf ../tensor_data/${PROJECT_NAME}
    rm -rf ../network/${PROJECT_NAME}
    rm -rf ../plot/${PROJECT_NAME}
fi
