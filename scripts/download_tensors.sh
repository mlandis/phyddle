#!/bin/sh

MODEL=$1
DIR="../workspace/tensor_data/${MODEL}"
./clean_project.sh ${MODEL}
mkdir -p ${DIR}
scp "$llserver_wd/projects/phyddle/tensor_data/${MODEL}/*.hdf5" ${DIR}
scp "$llserver_wd/projects/phyddle/tensor_data/${MODEL}/*.csv" ${DIR}


