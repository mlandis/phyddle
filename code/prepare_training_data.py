#!/usr/local/bin/python3

import glob
import os
import csv
import argparse
import shutil
from phyddle_util import init_process_settings


settings = {}
settings['model_name'] = 'bd1'
settings['prefix'] = 'sim'
settings = init_process_settings(settings)

prefix        = settings['prefix']
#model_dir     = '../model/' + settings['model_name']
raw_dir       = '../raw_data/' + settings['model_name'] #model_dir + '/data/raw'
train_dir     = '../tensor_data/' + settings['model_name'] #model_dir + '/data/formatted'

# make dir
os.makedirs(train_dir, exist_ok=True)

# collect files with replicate info
files = os.listdir(raw_dir)
info_files = [ x for x in files if 'info' in x ]

# sort replicate indices into size-category lists
size_sort = {}
for fn in info_files:
    fn = raw_dir + '/' + fn
    idx = -1
    size = -1
    with open(fn, newline='') as csvfile:
        info = csv.reader(csvfile, delimiter=',')
        for row in info:
            if row[0] == 'replicate_index':
                idx = int(row[1])
            elif row[0] == 'taxon_category':
                size = int(row[1])
            #print(', '.join(row))

    if size >= 0 and size not in size_sort:
        size_sort[size] = []
    if size >=0 and idx >= 0:
        size_sort[size].append(idx)

# build files
for k in sorted(list(size_sort.keys())):

    size_sort[k].sort()

    print('Formatting {n} files for taxon_category={i}'.format(n=len(size_sort[k]), i=k))
    
    out_cdvs_fn   = train_dir + '/' + prefix + '.nt' + str(k) + '.cdvs.data.csv'
    out_cblvs_fn  = train_dir + '/' + prefix + '.nt' + str(k) + '.cblvs.data.csv'
    out_stat_fn   = train_dir + '/' + prefix + '.nt' + str(k) + '.summ_stat.csv'
    out_labels_fn = train_dir + '/' + prefix + '.nt' + str(k) + '.labels.csv'
    out_info_fn   = train_dir + '/' + prefix + '.nt' + str(k) + '.info.csv'
    
    # cdv file tensor
    with open(out_cdvs_fn, 'w') as outfile:
        for i in size_sort[k]:
            fname = raw_dir + '/' + prefix + '.' + str(i) + '.cdvs.csv'
            with open(fname, 'r') as infile:
                s = infile.read()
                z = outfile.write(s)

    # cblvs tensor
    with open(out_cblvs_fn, 'w') as outfile:
        for i in size_sort[k]:
            fname = raw_dir + '/' + prefix + '.' + str(i) + '.cblvs.csv'
            with open(fname, 'r') as infile:
                s = infile.read()
                z = outfile.write(s)
    
    # summary stats tensor
    with open(out_stat_fn, 'w') as outfile:
        for j,i in enumerate(size_sort[k]):
            fname = raw_dir + '/' + prefix + '.' + str(i) + '.summ_stat.csv'
            with open(fname, 'r') as infile:
                if j == 0:
                    s = infile.read()
                    z = outfile.write(s)
                else:
                    s = ''.join(infile.readlines()[1:])
                    z = outfile.write(s)

    # labels input tensor
    with open(out_labels_fn, 'w') as outfile:
        for j,i in enumerate(size_sort[k]):
            fname = raw_dir + '/' + prefix + '.' + str(i) + '.param2.csv'
            with open(fname, 'r') as infile:
                if j == 0:
                    s = infile.read()
                    z = outfile.write(s)
                else:
                    s = ''.join(infile.readlines()[1:])
                    z = outfile.write(s)
            

