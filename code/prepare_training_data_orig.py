#!/usr/local/bin/python3

import glob
import os
import argparse
import shutil
from phyddle_util import init_process_settings


settings = {}
settings['model_name'] = 'geosse_v5'
settings['prefix'] = 'sim'
settings = init_process_settings(settings)

prefix        = settings['prefix']
#model_dir     = '../model/' + settings['model_name']
raw_dir       = '../raw_data/' + settings['model_name'] #model_dir + '/data/raw'
train_dir     = '../formatted_data/' + settings['model_name'] #model_dir + '/data/formatted'


# make dir
os.makedirs(train_dir, exist_ok=True)

# raw dirs for different num taxa
#os.listdir(r'C:\Users\enaknar\Desktop\pycharm')
raw_nt_dirs = []
for i in os.listdir(raw_dir):
    j = raw_dir + '/' + i
    if os.path.isdir(j):
        #print(i)
        raw_nt_dirs.append(i)


for i in raw_nt_dirs:

    out_cdvs_fn   = train_dir + '/' + prefix + '.' + i + '.cdvs.data.csv'
    out_cblvs_fn   = train_dir + '/' + prefix + '.' + i + '.cblvs.data.csv'
    out_labels_fn = train_dir + '/' + prefix + '.' + i + '.labels.csv'
    
    # raw files
    cdvs_files = glob.glob(raw_dir + '/' + i + '/*.cdvs.csv')
    cblvs_files = glob.glob(raw_dir + '/' + i + '/*.cblvs.csv')
    label_files = glob.glob(raw_dir + '/' + i + '/*.param2.csv')

    print('Formatting {n} files in {i}'.format(n=len(cdvs_files), i=i))

    # data input tensor
    with open(out_cdvs_fn, 'w') as outfile:
        for fname in cdvs_files:
            with open(fname, 'r') as infile:
                outfile.write(infile.read())
    
    with open(out_cblvs_fn, 'w') as outfile:
        for fname in cblvs_files:
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

