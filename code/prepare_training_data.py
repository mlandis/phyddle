#!/usr/local/bin/python3

import os

model_dir = '../model/geosse_v2/'
sim_dir = model_dir + 'data/raw'
train_dir = model_dir + 'data/train'
prefix = 'sim'

os.system('mkdir -p ' + train_dir)

# CBLVS input tensor
os.system('cat {sim_dir}/*.cblvs.* > {train_dir}/{prefix}.data.csv'.format( sim_dir=sim_dir, train_dir=train_dir, prefix=prefix ))

# param training labels
if False:
    label_cmd = 'tail -n1 {sim_dir}/*.param2.csv | grep , > {train_dir}/{prefix}.labels.csv'.format(sim_dir=sim_dir, train_dir=train_dir, prefix=prefix)
else: 
    label_cmd = ' \
tail -n1 {sim_dir}/*.param2.csv | grep , > {train_dir}/{prefix}.label_values.csv; \
cat {sim_dir}/*.param2.csv | head -n1 > {train_dir}/{prefix}.label_headers.csv; \
cat {train_dir}/{prefix}.label_*.csv > {train_dir}/{prefix}.labels.csv; \
rm {train_dir}/{prefix}.label_*.csv; \
'.format(sim_dir=sim_dir, train_dir=train_dir, prefix=prefix)
os.system(label_cmd)

