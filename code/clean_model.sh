#!/bin/sh
MODEL_NAME=$1

# only clean model if string not empty
if [ -n "${MODEL_NAME}" ]; then
    rm -rf ../raw_data/${MODEL_NAME}
    rm -rf ../tensor_data/${MODEL_NAME}
    rm -rf ../network/${MODEL_NAME}
    rm -rf ../plot/${MODEL_NAME}
fi
