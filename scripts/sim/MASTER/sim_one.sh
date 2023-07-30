#!/usr/bin/env sh

# get current working dir
CWD=`pwd`
cd sim/MASTER

# config filename
CFG_FN=$1

# sim prefix filename
TMP_FN=$2

echo "CFG_FN=${CFG_FN}"
echo "TMP_FN=${TMP_FN}"

# generate MASTER XML from analysis config
python gen_master_xml.py ${CFG_FN} ${TMP_FN}

# run MASTER against XML
#beast "${TMP_FN}.master.xml"

cd -
