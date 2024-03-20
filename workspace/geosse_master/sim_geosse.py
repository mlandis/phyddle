#!/usr/bin/env python3
import masterpy
import scipy as sp
import numpy as np
import sys
import os
import subprocess

# get arguments
out_path     = sys.argv[1]
prefix       = sys.argv[2]
idx          = int(sys.argv[3])
batch_size   = int(sys.argv[4])

# process arguments
tmp_fn       = f'{out_path}/{prefix}.{idx}'

# model setup
num_char = 3
num_states = 2
args = {
    'dir'                : out_path,        # dir for simulations
    'prefix'             : prefix,          # project name(s)
	'model_type'         : 'geosse',        # model type defines states & events
    'model_variant'      : 'equal_rates',   # model variant defines rates
    'num_char'           : num_char,        # number of evolutionary characters
    'num_states'         : num_states,      # number of states per character
    'sample_population'  : ['S'],           # name of population to sample
    'stop_time'          : 10,              # time to stop simulation
    'min_num_taxa'       : 10,              # min number of taxa for valid sim
    'max_num_taxa'       : 500,             # max number of taxa for valid sim
    'rv_fn'              : {                # distributions for model params
        'w': sp.stats.expon.rvs,
        'e': sp.stats.expon.rvs,
        'd': sp.stats.expon.rvs,
        'b': sp.stats.expon.rvs,
        'Stop_time': sp.stats.uniform.rvs,
        'nSampled_tips': sp.stats.randint.rvs
    },
    'rv_arg'             : {                # loc/scale for model param dists
        'w': { 'scale' : 0.2 },
        'e': { 'scale' : 0.1 },
        'd': { 'scale' : 0.1 },
        'b': { 'scale' : 0.5 },
        'Stop_time': { 'scale' : 10. },
        'nSampled_tips': { 'low': 10, 'high': 500 }
    }
}

# filesystem paths
xml_fn       = tmp_fn + '.xml'
param_mtx_fn = tmp_fn + '.param_col.csv'
param_vec_fn = tmp_fn + '.labels.csv'
phy_nex_fn   = tmp_fn + '.nex.tre'
dat_nex_fn   = tmp_fn + '.dat.nex'

# make sim dir for output
os.makedirs(out_path, exist_ok=True)

# load model
my_model = masterpy.load(args)

# assign index
my_model.set_model(idx)

# make XML
xml_str = my_model.make_xml(idx)
masterpy.write_to_file(xml_str, xml_fn)

# call BEAST
x = subprocess.run(['beast', xml_fn], capture_output=True)

# capture stdout/stderr since we're nesting subprocess calls
#x_stdout = x.stdout.decode('UTF-8')
#x_stderr = x.stderr.decode('UTF-8')
#sys.stdout.write(x_stdout)
#sys.stderr.write(x_stderr)

# get params (labels) from model
params = {
        'log10_w': np.log10(my_model.params['w']),
        'log10_e': np.log10(my_model.params['e']),
        'log10_d': np.log10(my_model.params['d']),
        'log10_b': np.log10(my_model.params['b'])
}
param_mtx_str,param_vec_str = masterpy.param_dict_to_str(params)

# save output
masterpy.write_to_file(param_mtx_str, param_mtx_fn)
masterpy.write_to_file(param_vec_str, param_vec_fn)

# convert phy.nex to dat.nex
int2vec = my_model.statespace.int2vec
nexus_str = masterpy.convert_phy2dat_nex_geosse(phy_nex_fn, int2vec)
masterpy.write_to_file(nexus_str, dat_nex_fn)

# log clean-up
#masterpy.cleanup(prefix=tmp_fn, clean_type)

# done!
quit()
