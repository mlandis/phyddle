#!/usr/local/bin/python3

# helper functions
from phyddle_util import *
#from model import Event, StateSpace, Model
from Model import *
from model_util import MasterXmlGenerator

# other dependencies
import numpy as np
import scipy as sp
import subprocess
import os
import argparse
import string
from itertools import chain, combinations
#import multiprocessing as mp
from joblib import Parallel, delayed
import time
from tqdm import tqdm
import cdvs_util

# start time
start = time.time()

# numpy printing format
np.set_printoptions(floatmode='unique', suppress=True)

#MyModel = SirmModel
MyModel = GeosseModel
my_model_args = {
    'num_locations' : 5,
    'model_variant' : 'equal_rates',
    'rv_fn' : { 'w': sp.stats.expon.rvs,
        'e': sp.stats.expon.rvs,
        'd': sp.stats.expon.rvs,
        'b': sp.stats.expon.rvs },
    'rv_arg' : {
        'w': { 'scale' : 2.0 },
        'e': { 'scale' : 0.5 },
        'd': { 'scale' : 1.0 },
        'b': { 'scale' : 3.0 }
    }
}

# default settings
settings = {}
settings['model_name']     = 'bd1'
#settings['model_type']     = ''
settings['start_idx']      = 0
settings['end_idx']        = 1 #99
settings['cfg_file']       = None # TODO: add config file parser
settings['use_parallel']   = False

# non-default settings passed by argsparse
settings     = init_sim_settings(settings)
model_name   = settings['model_name']
rep_idx      = np.arange( settings['start_idx'], settings['end_idx'] )
cfg_file     = settings['cfg_file']
use_parallel = settings['use_parallel']

print("Applied settings:")
print(settings)

# IO
out_dir    = '../raw_data/{model_name}'.format(model_name=model_name)
out_prefix = 'sim'
out_path   = out_dir + '/' + out_prefix

# init settings
num_rep        = len(rep_idx)
num_jobs       = -2
max_taxa       = [200, 500] #, 1000, 2000, 5000]

# make dirs
os.makedirs(out_dir, exist_ok=True)

# generate settings container
settings['max_taxa'] = np.max(max_taxa)

# main simulation function (looped)
def sim_one(k):

    # make filenames
    tmp_fn    = out_path + '.' + str(k)
    geo_fn    = tmp_fn + '.geosse.nex'
    tre_fn    = tmp_fn + '.tre'
    prune_fn  = tmp_fn + '.extant.tre'
    beast_fn  = tmp_fn + '.beast.log'
    xml_fn    = tmp_fn + '.xml'
    nex_fn    = tmp_fn + '.nex'
    json_fn   = tmp_fn + '.json'
    cblvs_fn  = tmp_fn + '.cblvs.csv'
    cdvs_fn   = tmp_fn + '.cdvs.csv'
    param1_fn = tmp_fn + '.param1.csv'
    param2_fn = tmp_fn + '.param2.csv'
    ss_fn     = tmp_fn + '.summ_stat.csv'
    info_fn   = tmp_fn + '.info.csv'
    
    # update settings
    settings['out_path'] = tmp_fn
    settings['replicate_index'] = k

    # instantiate model
    #mymodel = GeosseModel(num_locations=3, model_variant='equal_rates')
    #mymodel = SirmModel(num_locations=3, model_variant='equal_rates')
    mymodel = MyModel(**my_model_args)

    model_type = mymodel.model_type
    settings['model_type'] = model_type

    # get int/vec/str for states for some reason...
    #lbl2vec = mymodel.states.lbl2vec
    int2vec = mymodel.states.int2vec
    int2vecstr = mymodel.states.int2vecstr
    vecstr2int = mymodel.states.vecstr2int

    ## construct XML from model class
    xmlgen = MasterXmlGenerator(mymodel.df_events, mymodel.df_states, mymodel.settings)
    #print( xmlgen.make_reaction_vars() )
    # alternative, construct XML from event and states dataframes
    # xmlgen = MasterXmlGenerator(df_events, df_states)
    xmlgen.make_xml(newick_fn=tre_fn, nexus_fn=nex_fn, json_fn=json_fn)
    xml_str = xmlgen.xml_spec_str
    write_to_file(xml_str, xml_fn)

    # run BEAST/MASTER against XML
    beast_str = 'beast ' + xml_fn
    beast_out = subprocess.check_output(beast_str, shell=True, text=True, stderr=subprocess.STDOUT)
    write_to_file(beast_out, beast_fn)

    # verify tree size & existence!
    result_str = ''
    n_taxa_k = get_num_taxa(tre_fn, k, max_taxa)
    taxon_size_k = find_taxon_size(n_taxa_k, max_taxa)
    #print(n_taxa_k)

    # handle simulation based on tree size
    if n_taxa_k > np.max(max_taxa):
        # too many taxa
        result_str = '- replicate {k} simulated n_taxa={nt}'.format(k=k,nt=n_taxa_k)
        return result_str
    elif n_taxa_k <= 0:
        # no taxa
        result_str = '- replicate {k} simulated n_taxa={nt}'.format(k=k,nt=n_taxa_k)
        return result_str
    else:
        # valid number of taxa
        result_str = '+ replicate {k} simulated n_taxa={nt}'.format(k=k,nt=n_taxa_k)

        # generate extinct-pruned tree
        prune_success = make_prune_phy(tre_fn, prune_fn)

        # MJL 230411: probably too aggressive, should revisit
        if not prune_success:
            next

        # generate nexus file 0/1 ranges
        taxon_states,nexus_str = convert_nex(nex_fn, tre_fn, int2vec)
        write_to_file(nexus_str, geo_fn)

        # then get CBLVS working
        cblv,new_order = vectorize_tree(tre_fn, max_taxa=taxon_size_k, prob=1.0 )
        cblvs = make_cblvs_geosse(cblv, taxon_states, new_order)
       
        # NOTE: this if statement should not be needed, but for some reason the "next"
        # seems to run even when make_prune_phy returns False
        # generate CDVS file
        if prune_success:
            cdvs = cdvs_util.make_cdvs(prune_fn, taxon_size_k, taxon_states, int2vecstr) #states_bits_str)

        # output files
        mt_size   = cblv.shape[1]

    # record info
    info_str = settings_to_str(settings, mt_size)
    write_to_file(info_str, info_fn)

    # record labels (simulating parameters)
    param1_str,param2_str = param_dict_to_str(mymodel.rates)
    write_to_file(param1_str, param1_fn)
    write_to_file(param2_str, param2_fn)

    # record CBLVS data
    cblvs_str = np.array2string(cblvs, separator=',', max_line_width=1e200, threshold=1e200, edgeitems=1e200)
    cblvs_str = cblvs_str.replace(' ','').replace('.,',',').strip('[].') + '\n'
    write_to_file(cblvs_str, cblvs_fn)

    # record CDVS data
    if prune_success:
        cdvs = cdvs.to_numpy()
        cdvs_str = np.array2string(cdvs, separator=',', max_line_width=1e200, threshold=1e200, edgeitems=1e200)
        cdvs_str = cdvs_str.replace(' ','').replace('.,',',').strip('[].') + '\n'
        write_to_file(cdvs_str, cdvs_fn)

    # record summ stat data
    ss = make_summ_stat(tre_fn, geo_fn, vecstr2int)
    ss_str = make_summ_stat_str(ss)
    write_to_file(ss_str, ss_fn)

    # return status string
    return result_str

# dispatch jobs
if use_parallel:
    res = Parallel(n_jobs=num_jobs)(delayed(sim_one)(k) for k in tqdm(rep_idx))
else:
    res = [ sim_one(k) for k in tqdm(rep_idx) ]
        
# end time
end = time.time()
delta_time = np.round(end-start, decimals=3)
print('Elapsed time:', delta_time, 'seconds')

# done!
print('...done!')
quit()