#!/usr/bin/env python3
import master_util
import scipy as sp
import sys
import os
import subprocess

# get index for replicate
#tmp_fn = '/Users/mlandis/projects/phyddle/workspace/simulate/MASTER_example/sim.0'
print(sys.argv)
tmp_fn = sys.argv[1]
idx_str = tmp_fn.split('.')[-1]
idx = int(idx_str)

# model setup
num_char = 3
num_states = 2
args = {
    'sim_dir' : '/Users/mlandis/projects/phyddle/workspace/simulate', # main sim dir
	'proj'           : 'MASTER_example',             # project name(s)
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
        'b': sp.stats.expon.rvs
    },
    'rv_arg'             : {                # loc/scale for model param dists
        'w': { 'scale' : 0.2 },
        'e': { 'scale' : 0.1 },
        'd': { 'scale' : 0.1 },
        'b': { 'scale' : 0.5 }
    }
}

args['sim_proj_dir'] = f'{args["sim_dir"]}/{args["proj"]}'

# filesystem paths
sim_proj_dir = args['sim_proj_dir']
tmp_fn       = f'{sim_proj_dir}/sim.{idx}'
xml_fn       = tmp_fn + '.xml'
param_mtx_fn = tmp_fn + '.param_col.csv'
param_vec_fn = tmp_fn + '.param_row.csv'
phy_nex_fn   = tmp_fn + '.phy.nex'
dat_nex_fn   = tmp_fn + '.dat.nex'

# make sim dir for output
os.makedirs(sim_proj_dir, exist_ok=True)

# load model
my_model = master_util.load(args)

# assign index
my_model.set_model(idx)

# make XML
xml_str = my_model.make_xml(idx)

# get params (labels) from model
param_mtx_str,param_vec_str = master_util.param_dict_to_str(my_model.params)

# save output
master_util.write_to_file(xml_str, xml_fn)
master_util.write_to_file(param_mtx_str, param_mtx_fn)
master_util.write_to_file(param_vec_str, param_vec_fn)

# call BEAST
x = subprocess.run(['beast', xml_fn], capture_output=True)

# convert phy.nex to dat.nex
int2vec = my_model.states.int2vec
print(phy_nex_fn)
nexus_str = master_util.convert_phy2dat_nex(phy_nex_fn, int2vec)
master_util.write_to_file(nexus_str, dat_nex_fn)

# done!
quit()