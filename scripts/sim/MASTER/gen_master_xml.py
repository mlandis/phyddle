#!/usr/bin/env python

#import argparse
import master_util
import scipy as sp
import sys
import os

print(sys.argv)

idx = 0
num_char = 3
num_states = 2

# this could be loaded in elsewhere
args = {
    'sim_dir' : '~/projects/phyddle/workspace/simulate', # main sim dir
	'sim_proj'           : 'MASTER_example',             # project name(s)
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
    'rv_arg'             : {                # loc/scale/shape for model params
        'w': { 'scale' : 0.2 },
        'e': { 'scale' : 0.1 },
        'd': { 'scale' : 0.1 },
        'b': { 'scale' : 0.5 }
    }
}

args['sim_proj'] = f'{args["sim_dir"]}/{args["proj"]}'
print(args['sim_proj'])
os.makedirs(args['sim_proj'], exist_ok=True)

tmp_fn = args['sim_proj'] + '/sim.0'
xml_fn     = tmp_fn + '.xml'

# load model
my_model = master_util.load(args)

# assign index
my_model.set_model(idx)

# make XML
xml_str = my_model.make_xml(idx)

# save XML (can have model do it)
f = open(xml_fn, 'w')
f.write(xml_str)
f.close()

# save labels (can have model do it)
# ...

# done!
quit()
