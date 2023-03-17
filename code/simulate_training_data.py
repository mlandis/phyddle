#!/usr/local/bin/python3

# helper functions
from phyddle_util import *

# other dependencies
import numpy as np
import scipy as sp
import dendropy as dp
import subprocess
import os
import argparse
import string
from itertools import chain, combinations
from multiprocessing import Process

# NOTE: let's add a settings file  & parser
#       to specify state space, events, etc.

# NOTE: something like this for multiprocessing
# def main():
#    pool = mp.Pool(processes=8)
#        pool.map(parse_file, ['my_dir/' + filename for filename in os.listdir("my_dir")])
# or ...
#   pool = multiprocessing.Pool(4)
#   out1, out2, out3 = zip(*pool.map(calc_stuff, range(0, 10 * offset, offset))


# numpy printing format
np.set_printoptions(floatmode='unique', suppress=True)

# IO
out_dir    = '../model/geosse_v1/data/raw'
out_prefix = 'sim'
out_path   = out_dir + '/' + out_prefix
os.system('mkdir -p ' + out_dir)

# init settings
num_rep        = 100
max_taxa       = 999
num_regions    = 3
num_states     = 2**num_regions - 1
regions        = list(range(num_regions))
states         = list(powerset(regions))[1:]
states_str     = [ ''.join(list([string.ascii_uppercase[i] for i in s])) for s in states ]
states_str_inv = {}
states_inv     = {}
for i,v in enumerate(states_str):
    states_str_inv[v] = i
for i,v in enumerate(states):
    states_inv[v] = i
states_bits    = regions_to_binary(states, states_str, regions)

# model settings
model_type = 'iid'
num_feature_layers = 2
def rv_rate(size): return sp.stats.expon.rvs(size=size, loc=0., scale=0.1) 
def rv_effect(size): return sp.stats.norm.rvs(size=size, loc=0., scale=1.) 
def rv_feature(size): return sp.stats.gamma.rvs(size=size, loc=2., scale=2.) 

# generate settings container
settings = { 'max_taxa': max_taxa,
             'model_type': model_type,
             'num_feature_layers': num_feature_layers,
             'rv_rate': rv_rate,
             'rv_feature': rv_feature,
             'rv_effect': rv_effect }

# generate GeoSSE events
events = make_events(regions, states, states_inv)

#print('states ==> ', states, '\n')
#print('states_bits ==> ', states_bits, '\n')
#print('events ==>', events, '\n')

# simulate replicates
for k in range(num_rep):


    # update info for replicate
    settings['out_path'] = out_path+"."+str(k)
    geo_fn   = settings['out_path'] + '.geosse.nex'
    tre_fn   = settings['out_path'] + '.tre'
    nex_fn   = settings['out_path'] + '.nex'
    cblvs_fn = settings['out_path'] + '.cblvs.csv'
    param1_fn = settings['out_path'] + '.param1.csv'
    param2_fn = settings['out_path'] + '.param2.csv'
    beast_fn = settings['out_path'] + '.beast.log'
    xml_fn   = settings['out_path'] + '.xml'
    #out_fn   = settings['out_path']
    #cblv_fn  = settings['out_path'] + '.cblv.csv'

    # generate GeoSSE rates
    rates = make_rates(regions, states, events, settings)


    # generate MASTER XML string
    xml_str = make_xml(events, rates, states, states_str, settings)
    write_to_file(xml_str, xml_fn)

    # run BEAST/MASTER against XML
    beast_str = 'beast ' + xml_fn
    beast_out = subprocess.check_output(beast_str, shell=True, text=True, stderr=subprocess.STDOUT)
    write_to_file(beast_out, beast_fn)

    # verify tree size & existence!
    n_taxa_k = get_num_taxa(tre_fn, k, max_taxa)
    if n_taxa_k <= 0 or n_taxa_k >= max_taxa:
        continue

    # generate nexus file 0/1 ranges
    taxon_states = convert_geo_nex(nex_fn, tre_fn, geo_fn, states_bits)

    # encode dataset
    cblv,new_order = vectorize_tree(tre_fn, max_taxa=max_taxa, prob=1.0 )
    cblvs = make_cblvs_geosse(cblv, taxon_states, new_order)
    cblvs_str = np.array2string(cblvs, separator=',', max_line_width=1e200, threshold=1e200, edgeitems=1e200)
    cblvs_str = cblvs_str.replace(' ','').replace('.,',',').strip('[].') + '\n'
    write_to_file(cblvs_str, cblvs_fn)

    # record simulating parameters
    param1_str,param2_str = param_dict_to_str(rates)
    write_to_file(param1_str, param1_fn)
    write_to_file(param2_str, param2_fn)
   
    success_str = '+ replicate {k} simulated n_taxa={nt}'.format(k=k,nt=n_taxa_k)
    print(success_str)
    

# done!
print('...done!')
quit()
