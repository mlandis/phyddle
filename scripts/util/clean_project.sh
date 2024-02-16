#!/bin/sh
PROJECT_NAME=$1

# only clean project if string not empty
if [ -n "${PROJECT_NAME}" ]; then
    rm -rf ../workspace/${PROJECT_NAME}
fi
