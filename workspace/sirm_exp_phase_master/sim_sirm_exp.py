#!/usr/bin/env python3
import masterpy
import scipy as sp
import sys
import os
import subprocess
import json
import numpy as np
import copy

# get arguments
out_path = './sir_visit_approx'
prefix = 'out'
idx = 0
if len(sys.argv) >= 2:
    out_path     = sys.argv[1]
if len(sys.argv) >= 3:
    prefix       = sys.argv[2]
if len(sys.argv) >= 4:
    idx          = int(sys.argv[3])
if len(sys.argv) >= 5:
    batch_size   = int(sys.argv[4])
remove_big_files = True
if len(sys.argv) == 6:
    remove_big_files = sys.argv[5] == "True"


# model setup
args = {
    'dir'                : out_path,        # dir for simulations
    'prefix'             : prefix,          # project name(s)
    'model_type'         : 'sir',           # model type defines states & events
    # model variant defines rates
    'model_variant'      : [
                            'EqualRates',  # FreeRates, EqualRates
                            'Migration',  # [Migration, Visitor, VisitorApproximation]
                            'Stochastic'
                            ],           # [Stochastic, Deterministic]
    'num_char'           : 1,               # number of evolutionary characters 
    'num_states'         : 5,               # number of states per character
    'num_hidden_char'    : 1,               # number of hidden states
    'num_exposed_cat'    : 1,               # number of infected Exposed stages (>1)
#    'stop_time'          : None,            # time to stop simulation 
    'min_num_taxa'       : 20,             # min number of taxa for valid sim
    'max_num_taxa'       : 500,             # max number of taxa for valid sim
    'max_num_unsampled_lineages' : 100,      # max_num_taxa * max_num_unsampled_lineages == stopping condition
    'prop_extant_sampled' : 0.01,           # Expected proportion of lineages at stop_time to be sampled in tree
    'num_sample_time_pts' : 10,          # number of evenly spaced tim pts to sample population sizes
    'rv_fn'              : {                # distributions for model params
        'R0'                  : sp.stats.uniform.rvs,
        'Recover'             : sp.stats.uniform.rvs,
        'Sample'              : sp.stats.uniform.rvs,
        'Migrate'             : sp.stats.uniform.rvs,
        'S0'                  : sp.stats.uniform.rvs,
        'Stop_time'           : sp.stats.uniform.rvs,
        'nSampled_tips'       : sp.stats.randint.rvs,
        'Time_of_interest'    : sp.stats.norm.rvs
    },
    'rv_arg'                : {                # loc/scale/shape for param dists
        'R0'                : { 'loc' : 2,    'scale' : 6  }, 
        'Recover'           : { 'loc' : 10**-2,   'scale' : 4 * 10**-2    }, # 1 to 10 days, rate of 0.1 to 1
        'Sample'            : { 'loc' : 10**-4,   'scale' : 4.9 * 10**-3   }, # 10 to 1000 days, rate of 0.001 to 1
        'Migrate'           : { 'loc' : 10**-4, 'scale' : 4.9 * 10**-3 },
        'S0'                : { 'loc' : 10**6, 'scale' : 0  }, # 1000 to 10000 ind. in population
        'Stop_time'         : { 'loc' : 100,      'scale' : 0    },  # between 10 days and 2 months
        'nSampled_tips'     : { 'low' : 20.0,  'high' : 499.   },   # subsample samples
        'Time_of_interest' : { 'loc' : 0,    'scale' : 0} # standard deviation is 10 days
    }
}

# filesystem paths
tmp_fn       = out_path + "/" + prefix + '.' + str(idx)
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


# make stochastic files and gather more stats for labels
with open(phy_nex_fn) as file:
    nexus_tree_str = file.read()

num_tips = nexus_tree_str.count("Sampled")

#if nexus_tree_str.count("time") == 0:
if num_tips < 20:
    os.remove(dat_json_fn)
    os.remove(phy_nex_fn)
    os.remove(phy_nwk_fn)
    os.remove(xml_fn)
    quit() #sys.exit("no tree")

# include sim stats such as prevalence at time pt of interest and 
# cumulative number of samples up to present
sim_stats = my_model.get_json_stats(dat_json_fn)
most_recent_tip_age = masterpy.get_age_most_recent_tip(nexus_tree_str, sim_stats['actual_sim_time'])
if sim_stats['total_sampled'][0] > 0:
    proportion_sample_in_tree =  [nexus_tree_str.count("Sampled") / sim_stats['total_sampled'][0]]
else:
    proportion_sample_in_tree = [0.]

# if no extant samples then reject sim by exiting
#print(my_model.params['Stop_time'], sim_stats['actual_sim_time'], most_recent_tip_age)
if my_model.params['Stop_time'][0] != sim_stats['actual_sim_time'][0] or most_recent_tip_age != 0:
    os.remove(dat_json_fn)
    os.remove(phy_nex_fn)
    os.remove(phy_nwk_fn)
    os.remove(xml_fn)
    quit() #sys.exit("no extant")

phy_state_dat = masterpy.convert_phy2dat_nex(nexus_tree_str, my_model.num_states)
masterpy.write_to_file(phy_state_dat, dat_nex_fn)

if remove_big_files:
    os.remove(phy_nex_fn)

# gather all data for labels files
params_and_popstats = {
                        **my_model.params, 
                        **sim_stats,
                        'most_recent_tip_age': most_recent_tip_age,
                        'proportion_sample_in_tree' : proportion_sample_in_tree
                        }

params_and_popstats = masterpy.log_params(params_and_popstats, 
                                          ['R0', 
                                           'Sample', 
                                           'Recover', 
                                           'Migrate',
                                           'Infect'])

param_mtx_str, param_vec_str = masterpy.param_dict_to_str(params_and_popstats)

# make label file
masterpy.write_to_file(param_mtx_str, param_mtx_fn)
masterpy.write_to_file(param_vec_str, param_vec_fn)

# delete json file
if remove_big_files:
    os.remove(dat_json_fn)

quit()
