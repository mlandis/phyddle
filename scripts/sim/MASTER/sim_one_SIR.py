#!/usr/bin/env python3
import master_util
import scipy as sp
import sys
import os
# import re
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
args = {
    'sim_dir'            : sim_dir,         # dir for simulations
    'proj'               : proj,            # project name(s)
	'model_type'         : 'sirm',          # model type defines states & events
    'model_variant'      : 'equal_rates',   # model variant defines rates
    'num_char'           : 1,               # number of evolutionary characters
    'num_states'         : 3,               # number of states per character
    'sample_population'  : ['S'],           # name of population to sample
    'stop_time'          : 10,              # time to stop simulation
    'min_num_taxa'       : 10,              # min number of taxa for valid sim
    'max_num_taxa'       : 500,             # max number of taxa for valid sim
    'rv_fn'              : {                # distributions for model params
        'R0'             : sp.stats.uniform.rvs,
        'recovery'       : sp.stats.uniform.rvs,
        'sampling'       : sp.stats.uniform.rvs,
        'migration'      : sp.stats.uniform.rvs,
        'S0'             : sp.stats.uniform.rvs
    },
    'rv_arg'             : {                # loc/scale/shape for param dists
        'R0'             : { 'loc' : 1.,    'scale' : 9.    },
        'recovery'       : { 'loc' : 0.01,  'scale' : 0.09  },
        'sampling'       : { 'loc' : 0.1,   'scale' : 0.9   },
        'migration'      : { 'loc' : 0.1,   'scale' : 0.9   },
        'S0'             : { 'loc' : 1000., 'scale' : 4000. }
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

# capture stdout/stderr since we're nesting subprocess calls
# x_stdout = x.stdout.decode('UTF-8')
# x_stderr = x.stderr.decode('UTF-8')
# sys.stdout.write(x_stdout)
# sys.stderr.write(x_stderr)

# convert phy.nex to dat.nex
int2vec = my_model.states.int2vec
nexus_str = master_util.convert_phy2dat_nex(phy_nex_fn, int2vec)
master_util.write_to_file(nexus_str, dat_nex_fn)

# log clean-up
#master_util.cleanup(prefix=tmp_fn, clean_type)

# done!
quit()
