#!/bin/sh

MODEL=$1
DIR="../workspace/network/${MODEL}"
./clean_project.sh ${MODEL}
mkdir -p ${DIR}
scp "$llserver_wd/projects/phyddle/network/${MODEL}/sim_batchsize*_numepoch*nt*.*" ${DIR}


