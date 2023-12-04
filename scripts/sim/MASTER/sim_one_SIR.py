#!/usr/bin/env python3
import masterpy
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

# NUMBER OF LOCATIONS/POPULATIONS
num_states = 2

args = {
    'sim_dir'            : sim_dir,         # dir for simulations
    'proj'               : proj,            # project name(s)
	'model_type'         : 'sir',           # model type defines states & events
    # model variant defines rates
    'model_variant'      : ['EqualRates',   
                            'Exposed',
                            'Visitor'],     
    'num_char'           : 2,               # number of evolutionary characters
    'num_states'         : num_states,      # number of states per character
    'num_hidden_char'    : 1,               # number of hidden states
    'num_exposed_cat'    : 2,               # number of infected Exposed stages (>1)
    'stop_time'          : 1000.0,           # time to stop simulation
    'min_num_taxa'       : 1,               # min number of taxa for valid sim
    'max_num_taxa'       : 500,             # max number of taxa for valid sim
    'rv_fn'              : {                # distributions for model params
        'R0'             : sp.stats.uniform.rvs,
        'Recover'        : sp.stats.uniform.rvs,
        'Sample'         : sp.stats.uniform.rvs,
        'ProgressInfected' : sp.stats.uniform.rvs,
        'VisitDepart'    : sp.stats.uniform.rvs,
        'VisitReturn'    : sp.stats.uniform.rvs,
        'S0'             : sp.stats.uniform.rvs,
        'V0'             : sp.stats.uniform.rvs
    },
    'rv_arg'                : {                # loc/scale/shape for param dists
        'R0'                : { 'loc' : 1.,     'scale' : 9.     }, 
        'Recover'           : { 'loc' : 0.1,    'scale' : 0.9    }, # 1 to 10 days, rate of 0.1 to 1
        'Sample'            : { 'loc' : 0.1,    'scale' : 0.9    }, # 1 to 10 days, rate of 0.1 to 1
        'ProgressInfected'  : { 'loc' : 0.01,   'scale' : 0.09   }, # 0.1 to 1 day,  rate of 1 to 10
        'VisitDepart'       : { 'loc' : 0.001,  'scale' : 0.099  }, # 1 to 100 days, rate of 0.01 to 1
        'VisitReturn'       : { 'loc' : 0.01,   'scale' : 0.99   }, # 0.1 to 10 days, rate of 0.1 to 10
        'S0'                : { 'loc' : 1000., 'scale' : 9000. }, # 10000 to 100000 ind. in population
        'V0'                : { 'loc' : 0.001,  'scale' : 0.009  }  # 0.1 to 1% of individuals away?
    }
}

#'R0'                : 1 to 10
#'Recover'           : 1 to 10 days,  rate of 0.1 to 1
#'Sample'            : 1 to 10 days,  rate of 0.1 to 1
#'ProgressInfected'  : 0.1 to 1 day,  rate of 1 to 10
#'VisitDepart'       : 1 to 100 days, rate of 0.01 to 1
#'VisitReturn'       : 0.1 to 10 days, rate of 0.1 to 10
#'S0'                : 10000 to 100000 ind. in population
#'V0'                : 0.1 to 1% of individuals away?

# filesystem paths
xml_fn       = tmp_fn + '.xml'
param_mtx_fn = tmp_fn + '.param_col.csv'
param_vec_fn = tmp_fn + '.labels.csv'
phy_nex_fn   = tmp_fn + '.nex.tre'
dat_nex_fn   = tmp_fn + '.dat.nex'

# make sim dir for output
os.makedirs(sim_proj_dir, exist_ok=True)

# load model
my_model = masterpy.load(args)

# assign index
my_model.set_model(idx)

# print out model parameters
# NOTE: add print summary to masterpy
params = my_model.params
for k,v in params.items():
    print(k, '\t', v)

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
# x_stdout = x.stdout.decode('UTF-8')
# x_stderr = x.stderr.decode('UTF-8')
# sys.stdout.write(x_stdout)
# sys.stderr.write(x_stderr)

# convert phy.nex to dat.nex
#int2vec = my_model.states.int2vec
print(my_model.states)
states = [num_states]
states = my_model.states['Sampled'][0:-1] # drop hidden state
print(states)
nexus_str = masterpy.convert_phy2dat_nex(phy_nex_fn, states=states)
masterpy.write_to_file(nexus_str, dat_nex_fn)

# log clean-up
#masterpy.cleanup(prefix=tmp_fn, clean_type)

# done!
quit()
