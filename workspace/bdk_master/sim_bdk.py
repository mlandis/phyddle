#!/usr/bin/env python3
import masterpy
import scipy as sp
import sys
import os
import subprocess
import json
import numpy as np
import dendropy as dp

# get arguments
out_path     = sys.argv[1]
prefix       = sys.argv[2]
idx          = int(sys.argv[3])
batch_size   = int(sys.argv[4])

# process arguments
tmp_fn       = f'{out_path}/{prefix}.{idx}'

# model setup
args = {
    'dir'                : out_path,         # dir for simulations
    'prefix'             : prefix,            # project name(s)
	'model_type'         : 'birthdeath',           # model type defines states & events
    # model variant defines rates
    'model_variant'      : ['FreeRates',            # [FreeRates, EqualRates]
                            'DensityDependentDeath'
                            ],
    'num_char'           : 0,               # number of evolutionary characters 
    'num_states'         : 1,               # number of states per character
    'num_hidden_char'    : 1,               # number of hidden states
    'stop_time'          : None,            # time to stop simulation 
    'min_num_taxa'       : 5,               # min number of taxa for valid sim
    'max_num_taxa'       : 500,             # max number of taxa for valid sim
    'rv_fn'              : {                # distributions for model params
        'DivConst'       : sp.stats.expon.rvs,
        'Turnover'       : sp.stats.beta.rvs,
        'DeathCarryK'    : sp.stats.uniform.rvs,
        'Stop_time'      : sp.stats.uniform.rvs,
        'nSampled_tips'  : sp.stats.randint.rvs
    },
    'rv_arg'                : {                # loc/scale/shape for param dists
        'DivConst'          : { 'loc'   : 0.0,     'scale' : 5.00     }, 
        'Turnover'          : { 'a'     : 1.0,     'b'     : 1.0      }, 
        'DeathCarryK'       : { 'scale' : 10,      'scale' : 1990     }, 
        'Stop_time'         : { 'loc'   : 5.0,     'scale' : 0.0     }, 
        'nSampled_tips'     : { 'low'   : 49999,     'high'  : 50000      }  
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
my_model.set_model(idx)

# make XML
xml_str = my_model.make_xml(idx)

# save xml output
masterpy.write_to_file(xml_str, xml_fn)

# call BEAST
x = subprocess.run(['beast', xml_fn], capture_output=True)

#masterpy.make_extant_phy(phy_nwk_fn)

# create dat.phy.nex file if stochastic sim
#if my_model.model_stochastic:
#    phy_state_dat = masterpy.convert_phy2dat_nex(phy_nex_fn, my_model.num_states)
#    masterpy.write_to_file(phy_state_dat, dat_nex_fn)

# include sim stats such as prevalence at time pt of interest and 
# cumulative number of samples
param_mtx_str, param_vec_str = masterpy.param_dict_to_str( my_model.params )

# make label file
masterpy.write_to_file(param_mtx_str, param_mtx_fn)
masterpy.write_to_file(param_vec_str, param_vec_fn)

# remove stem branch from newick tree
if my_model.model_stochastic:   
    masterpy.remove_stem_branch(phy_nwk_fn)

# print("finished sim " + str(idx))
quit()
