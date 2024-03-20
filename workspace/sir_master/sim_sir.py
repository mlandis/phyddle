#!/usr/bin/env python3
import masterpy
import pandas as pd
import numpy as np
import scipy as sp
import sys
import os
import subprocess
import json

# get arguments
out_path     = sys.argv[1]
prefix       = sys.argv[2]
idx          = int(sys.argv[3])
batch_size   = int(sys.argv[4])

# process arguments
tmp_fn       = f'{out_path}/{prefix}.{idx}'

# model setup
args = {
    'dir'                : out_path,        # dir for simulations
    'prefix'             : prefix,          # file prefix
	'model_type'         : 'sir',           # model type defines states & events
    # model variant defines rates
    'model_variant'      : ['EqualRates',            # [FreeRates, EqualRates]
                            'Stochastic'],           # [Stochastic, Deterministic]
    'num_char'           : 1,               # number of evolutionary characters 
    'num_states'         : 1,               # number of states per character
    'num_hidden_char'    : 1,               # number of hidden states
    'num_exposed_cat'    : 1,               # number of infected Exposed stages (>1)
    'stop_time'          : None,            # time to stop simulation 
    'min_num_taxa'       : 10,               # min number of taxa for valid sim
    'max_num_taxa'       : 500,             # max number of taxa for valid sim
    'rv_fn'              : {                # distributions for model params
        'R0'             : sp.stats.uniform.rvs,
        'Recover'        : sp.stats.uniform.rvs,
        'Sample'         : sp.stats.uniform.rvs,
        'S0'             : sp.stats.uniform.rvs,
        'Stop_time'      : sp.stats.uniform.rvs,
        'nSampled_tips'  : sp.stats.randint.rvs,
        'Time_before_present' : sp.stats.expon.rvs
    },
    'rv_arg'                : {                # loc/scale/shape for param dists
        'R0'                : { 'loc' : 1.0,     'scale' : 7.0   }, 
        'Recover'           : { 'loc' : 0.1,     'scale' : 0.9   }, # 1 to 10 days, rate of 0.1 to 1
        'Sample'            : { 'loc' : 0.1,     'scale' : 0.9   }, # 1 to 100 days, rate of 0.1 to 1
        'S0'                : { 'loc' : 10000.,   'scale' : 90000. }, # 1000 to 10000 ind. in population
        'Stop_time'         : { 'loc' : 20,      'scale' : 200   },  # between 10 days and 1 year
        'nSampled_tips'     : { 'low' : 50.0,    'high' : 450.   },   # subsample samples
        'Time_before_present' : { 'loc' : 0,     'scale' : 30}
    }
}

# filesystem paths
xml_fn       = tmp_fn + '.xml'
param_mtx_fn = tmp_fn + '.param_col.csv'
param_vec_fn = tmp_fn + '.labels.csv'
phy_nex_fn   = tmp_fn + '.nex.tre'
phy_nwk_fn   = tmp_fn + '.tre'
dat_nex_fn   = tmp_fn + '.dat.nex'
dat_json_fn  = tmp_fn + '.json'

# make sim dir for output
os.makedirs(out_path, exist_ok=True)

# load model
my_model = masterpy.load(args)

# NOTE: .set_model is called in the constructor.
# only using here to set the seed for validation:
# my_model.set_model(idx)

# make XML
xml_str = my_model.make_xml(idx)

# save xml output
masterpy.write_to_file(xml_str, xml_fn)

# call BEAST
x = subprocess.run(['beast', xml_fn], capture_output=True)

# include sim stats such as prevalence at time pt of interest and 
# cumulative number of samples up to present
# sim_stats = my_model.get_json_stats(dat_json_fn)

# make stochastic files and gather more stats for labels
if my_model.model_stochastic and os.path.isfile(phy_nwk_fn):
    phy_state_dat = masterpy.blank_phy2dat_nex(phy_nwk_fn)
    masterpy.write_to_file(phy_state_dat, dat_nex_fn)
    # masterpy.remove_stem_branch(phy_nwk_fn)

# gather all data for labels files
params = {
        'log10_R0'      : np.log10(my_model.params['R0']),
        'log10_Sample'  : np.log10(my_model.params['Sample']),
        'log10_Infect'  : np.log10(my_model.params['Infect']),
        'log10_Recover' : np.log10(my_model.params['Recover']),
        'log10_S0'      : np.log10(my_model.params['S0'])
}

param_mtx_str, param_vec_str = masterpy.param_dict_to_str( params )

# make label file
masterpy.write_to_file(param_mtx_str, param_mtx_fn)
masterpy.write_to_file(param_vec_str, param_vec_fn)

# delete json file
# os.remove(dat_json_fn)

quit()
