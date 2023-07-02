#!/bin/sh

MODEL=$1
DIR="../workspace/plot/${MODEL}"
./clean_project.sh ${MODEL}
mkdir -p ${DIR}
scp "$llserver_wd/projects/phyddle/plot/${MODEL}/*.pdf" ${DIR}


