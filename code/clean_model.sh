#!/bin/sh
MODEL_NAME=$1
rm -rf ../raw_data/${MODEL_NAME}
rm -rf ../formatted_data/${MODEL_NAME}
rm -rf ../network/${MODEL_NAME}
rm -rf ../plot/${MODEL_NAME}
