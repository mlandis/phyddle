#!/usr/local/bin/python3

import glob
import os
import argparse
import shutil
from phyddle_util import init_process_settings


settings = {}
settings['model_name'] = 'geosse_share_v1'
settings = init_process_settings(settings)

model_dir = '../model/' + settings['model_name']
raw_dir = model_dir + '/data/raw'
train_dir = model_dir + '/data/train'
prefix = 'sim'
out_data_fn = train_dir + '/' + prefix + '.data.csv'
out_labels_fn = train_dir + '/' + prefix + '.labels.csv'

# make dir
if not os.path.exists(train_dir):
    os.makedirs(train_dir)

# raw files
data_files = glob.glob(raw_dir + '/*.cblvs.csv')
label_files = glob.glob(raw_dir + '/*.param2.csv')

# data input tensor
with open(out_data_fn, 'w') as outfile:
    for fname in data_files:
        with open(fname, 'r') as infile:
            outfile.write(infile.read())

# labels input tensor
with open(out_labels_fn, 'w') as outfile:
    for i,fname in enumerate(label_files):
        with open(fname, 'r') as infile:
            if i == 0:
                outfile.write(infile.read())
            else:
                s = ''.join(infile.readlines()[1:])
                outfile.write(s)

