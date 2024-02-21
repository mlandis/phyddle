#!/usr/bin/env python3
import masterpy
import scipy as sp
import sys
import os
import subprocess

# get arguments
out_path     = sys.argv[1]
idx          = int(sys.argv[2])
batch_size   = int(sys.argv[3])

# process arguments
tmp_fn       = sys.argv[1] + f'/sim.{idx}'
sim_tok      = tmp_fn.split('/')
sim_dir      = '/'.join(sim_tok[:-2])
proj         = sim_tok[-2]
sim_proj_dir = f'{sim_dir}/{proj}'

# model setup
num_char = 3
num_states = 2
args = {
    'sim_dir'            : sim_dir,         # dir for simulations
    'proj'               : proj,            # project name(s)
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

# filesystem paths
xml_fn       = tmp_fn + '.xml'
param_mtx_fn = tmp_fn + '.param_col.csv'
param_vec_fn = tmp_fn + '.param_row.csv'
phy_nex_fn   = tmp_fn + '.phy.nex'
dat_nex_fn   = tmp_fn + '.dat.nex'

# make sim dir for output
os.makedirs(sim_proj_dir, exist_ok=True)

# load model
my_model = masterpy.load(args)

# assign index
my_model.set_model(idx)

# make XML
xml_str = my_model.make_xml(idx)

# get params (labels) from model
param_mtx_str,param_vec_str = masterpy.param_dict_to_str(my_model.params)

# save output
masterpy.write_to_file(xml_str, xml_fn)
masterpy.write_to_file(param_mtx_str, param_mtx_fn)
masterpy.write_to_file(param_vec_str, param_vec_fn)

# call BEAST
x = subprocess.run(['beast', xml_fn], capture_output=True)

# capture stdout/stderr since we're nesting subprocess calls
#x_stdout = x.stdout.decode('UTF-8')
#x_stderr = x.stderr.decode('UTF-8')
#sys.stdout.write(x_stdout)
#sys.stderr.write(x_stderr)

# convert phy.nex to dat.nex
int2vec = my_model.states.int2vec
nexus_str = masterpy.convert_phy2dat_nex(phy_nex_fn, int2vec)
masterpy.write_to_file(nexus_str, dat_nex_fn)

# log clean-up
#masterpy.cleanup(prefix=tmp_fn, clean_type)

# done!
quit()
