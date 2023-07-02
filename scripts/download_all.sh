#!/bin/sh

MODEL=$1

LOCAL_TENSOR_FP="../workspace/tensor_data/${MODEL}"
LOCAL_NETWORK_FP="../workspace/network/${MODEL}"
LOCAL_PLOT_FP="../workspace/plot/${MODEL}"

REMOTE_TENSOR_FP="$llserver_wd/projects/phyddle/tensor_data/${MODEL}/*.hdf5" 
REMOTE_NETWORK_FP="$llserver_wd/projects/phyddle/network/${MODEL}/sim_batchsize*_numepoch*_nt*.*" 
REMOTE_PLOT_FP="$llserver_wd/projects/phyddle/plot/${MODEL}/*.pdf" 

mkdir -p ${LOCAL_TENSOR_FP}
mkdir -p ${LOCAL_NETWORK_FP}
mkdir -p ${LOCAL_PLOT_FP}

ssh -fMNS bgconn -o ControlPersist=yes $llserver
scp -o ControlPath=bgconn ${REMOTE_TENSOR_FP} ${LOCAL_TENSOR_FP}
scp -o ControlPath=bgconn ${REMOTE_NETWORK_FP} ${LOCAL_NETWORK_FP}
scp -o ControlPath=bgconn ${REMOTE_PLOT_FP} ${LOCAL_PLOT_FP}
ssh -S bgconn -O exit -
