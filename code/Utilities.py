# general libraries
import argparse
import importlib
#import sys
import re
import os
#import itertools
#import dill
#import random

# Call before importing Tensorflow to suppress INFO messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

import pandas as pd
import numpy as np
import dendropy as dp
from itertools import chain, combinations
from keras import backend as K    # <- move this into Learning if possible

#import scipy as sp
#import matplotlib.pyplot as plt
#import matplotlib.ticker as ticker
#import tensorflow as tf
#from sklearn.preprocessing import StandardScaler
#from sklearn.decomposition import PCA
#from sklearn import metrics
#from collections import Counter
#from ete3 import Tree
#from scipy.interpolate import RegularGridInterpolator
#from sklearn.neighbors import KDTree


NUM_DIGITS = 10
np.set_printoptions(floatmode='maxprec', precision=NUM_DIGITS)
pd.set_option('display.precision', NUM_DIGITS)
pd.set_option('display.float_format', lambda x: f'{x:,.3f}')

#max_len = 501
# TURN_ONE = 'turn_one'

# the information on state is saved as 't_s' in the newick tree
# T_S = 't_s'
# STATE = 'state'
# DIVERSIFICATION_SCORE = 'diversification_score'

# sys.setrecursionlimit(100000)

##################
# Helper Classes #
##################

# model events
class Event:
    # initialize
    def __init__(self, idx, r=0.0, n=None, g=None, ix=None, jx=None):
        self.i = -1
        self.j = -1
        self.k = -1
        self.idx = idx
        if 'i' in idx:
            self.i = idx['i']
        if 'j' in idx:
            self.j = idx['j']
        if 'k' in idx:
            self.k = idx['k']
        self.rate = r
        self.name = n
        self.group = g
        self.ix = ix
        self.jx = jx
        self.reaction = ' + '.join(ix) + ' -> ' + ' + '.join(jx)
        
    # make print string
    def make_str(self):
        s = 'Event({name},{group},{rate},{idx})'.format(name=self.name, group=self.group, rate=self.rate, idx=self.idx)        
        #s += ')'
        return s
    # representation string
    def __repr__(self):
        return self.make_str()
    # print string
    def __str__(self):
        return self.make_str()


# state space
class States:
    def __init__(self, lbl2vec):

        # state space dictionary (input)
        self.lbl2vec      = lbl2vec

        # basic info
        self.int2lbl        = list( lbl2vec.keys() )
        self.int2vec        = list( lbl2vec.values() )
        self.int2int        = list( range(len(self.int2vec)) )
        self.int2set        = list( [ tuple([y for y,v in enumerate(x) if v == 1]) for x in self.int2vec ] )
        self.lbl_one        = list( set(''.join(self.int2lbl)) )
        self.num_char       = len( self.int2vec[0] )
        self.num_states     = len( self.lbl_one )

        # relational info
        self.lbl2int = {k:v for k,v in list(zip(self.int2lbl, self.int2int))}
        self.lbl2set = {k:v for k,v in list(zip(self.int2lbl, self.int2set))}
        self.lbl2vec = {k:v for k,v in list(zip(self.int2lbl, self.int2vec))}
        self.vec2int = {tuple(k):v for k,v in list(zip(self.int2vec, self.int2int))}
        self.vec2lbl = {tuple(k):v for k,v in list(zip(self.int2vec, self.int2lbl))}
        self.vec2set = {tuple(k):v for k,v in list(zip(self.int2vec, self.int2set))}
        self.set2vec = {tuple(k):v for k,v in list(zip(self.int2set, self.int2vec))}
        self.set2int = {tuple(k):v for k,v in list(zip(self.int2set, self.int2int))}
        self.set2lbl = {tuple(k):v for k,v in list(zip(self.int2set, self.int2lbl))}
        self.int2vecstr = [ ''.join([str(y) for y in x]) for x in self.int2vec ]
        self.vecstr2int = { v:i for i,v in enumerate(self.int2vecstr) }
       
    def make_str(self):
        # state space: {'A': [1, 0, 0], 'B': [0, 1, 0], 'C': [0, 0, 1], 'AB': [1, 1, 0], 'AC': [1, 0, 1], 'BC': [0, 1, 1], 'ABC': [1, 1, 1]}
        # string: Statespace(A,0,100;B,1,010;C,2,001;AB,3,110;AC,4,101;BC,5,011;ABC,6,111)
        s = 'Statespace('
        x = []
        for i in self.int2int:
            # each state in the space is reported as STRING,INT,VECTOR;
            x.append( self.int2lbl[i] + ',' + str(self.int2int[i]) + ',' + ''.join( str(x) for x in self.int2vec[i]) )
        s += ';'.join(x) + ')'
        return s

    # representation string
    def __repr__(self):
        return self.make_str()
    # print string
    def __str__(self):
        return self.make_str()
    
    def make_df(self):
        df = pd.DataFrame()



#-----------------------------------------------------------------------------------------------------------------#

###################
# CONFIG LOADER   #
###################

def load_config(config_fn, arg_overwrite=True):
    
    # KEEP THIS: Want to improve precedence so CLI-provided-arg > CFG-arg > CLI-default-arg

    # # argument parsing
    # parser = argparse.ArgumentParser(description='phyddle pipeline config', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('-c', '--cfg',          dest='config_fn', type=str, default='config', help='Config file name')
    # parser.add_argument('-f', '--force',        action='store_true', help='Arguments override config file settings')
    # parser.add_argument('--proj',               dest='proj', type=str, default='my_project', help='Project name used as directory across pipeline stages')
    # parser.add_argument('--use_parallel',       dest='use_parallel', type=bool, default=True, help='Use parallelization? (recommended)')
    # parser.add_argument('--num_proc',           dest='num_proc', type=int, default=-2, help='How many cores for multiprocessing? (e.g. 4 uses 4, -2 uses all but 2)')
    # # directory settings
    # parser.add_argument('--sim_dir',            dest='sim_dir', type=str, default='../raw_data', help='Directory for raw simulated data')
    # parser.add_argument('--fmt_dir',            dest='fmt_dir', type=str, default='../tensor_data', help='Directory for tensor-formatted simulated data')
    # parser.add_argument('--net_dir',            dest='net_dir', type=str, default='../network', help='Directory for trained networks and predictions')
    # parser.add_argument('--plt_dir',            dest='plt_dir', type=str, default='../plot', help='Directory for plotted results')
    # parser.add_argument('--pred_dir',           dest='pred_dir', type=str, help='Predict results for dataset located in this directory')
    # # model settings
    # #parser.add_argument('--show_models',        dest='show_models', type=bool, default=False, help='Print all available model types and variants?')
    # parser.add_argument('--model_type',         dest='model_type', type=str, help='Model type')
    # parser.add_argument('--model_variant',      dest='model_variant', type=str, help='Model variant')
    # parser.add_argument('--num_char',           dest='num_char', type=int, help='Number of characters')
    # # simulation settings
    # parser.add_argument('--sim_logging',        dest='sim_logging', type=str, default='verbose', choices=['clean', 'verbose', 'compress'], help='Simulation logging style')
    # parser.add_argument('--start_idx',          dest='start_idx', type=int, default=0, help='Start index for simulation')
    # parser.add_argument('--end_idx',            dest='end_idx', type=int, default=100, help='End index for simulation')
    # parser.add_argument('--stop_time',          dest='stop_time', type=float, default=10.0, help='Maximum duration of evolution for each simulation')
    # parser.add_argument('--stop_floor_sizes',   dest='stop_floor_sizes', type=int, default=0, help='Minimum number of taxa for each simulation')
    # parser.add_argument('--stop_ceil_sizes',    dest='stop_ceil_sizes', type=int, default=500, help='Maximum number of taxa for each simulation')
    # # formatting settings
    # parser.add_argument('--tensor_format',      dest='tensor_format', type=str, default='hdf5', choices=['hdf5', 'csv'], help='Storage format for simulation tensors')
    # parser.add_argument('--tree_type',          dest='tree_type', type=str, choices=['extant', 'serial'], help='Type of tree')
    # # learning settings
    # parser.add_argument('--tree_size',          dest='tree_size', type=int, help='Number of taxa in phylogenetic tensor')
    # parser.add_argument('--num_epochs',         dest='num_epochs', type=int, default=21, help='Number of learning epochs')
    # parser.add_argument('--batch_size',         dest='batch_size', type=int, default=128, help='Training batch sizes during learning')
    # parser.add_argument('--prop_test',          dest='prop_test', type=float, default=0.05, help='Proportion of data used as test examples (demonstrate trained network performance)')
    # parser.add_argument('--prop_validation',    dest='prop_validation', type=float, default=0.05, help='Proportion of data used as validation examples (diagnose network overtraining)')
    # parser.add_argument('--prop_calibration',   dest='prop_calibration', type=float, default=0.20, help='Proportion of data used as calibration examples (calibrate conformal prediction intervals)')
    # parser.add_argument('--alpha_CQRI',         dest='alpha_CQRI', type=float, default=0.95, help='Expected coverage percent for prediction intervals')
    # parser.add_argument('--loss',               dest='loss', type=str, default='mse', help='Loss function used as optimization criterion')
    # parser.add_argument('--optimizer',          dest='optimizer', type=str, default='adam', help='Method used for optimizing neural network')
    # # plotting settings
    # parser.add_argument('--network_prefix',     dest='network_prefix', type=str, help='Plot results related to this network prefix')
    # # prediction settings
    # parser.add_argument('--pred_prefix',        dest='pred_prefix', type=str, help='Predict results for this dataset')
    
    # argument parsing
    parser = argparse.ArgumentParser(description='phyddle pipeline config',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-c', '--cfg',          dest='config_fn', type=str, help='Config file name')
    #parser.add_argument('-f', '--force',        action='store_true', help='Arguments override config file settings')
    parser.add_argument('--proj',               dest='proj', type=str, help='Project name used as directory across pipeline stages')
    parser.add_argument('--use_parallel',       dest='use_parallel', type=bool, help='Use parallelization? (recommended)')
    parser.add_argument('--num_proc',           dest='num_proc', type=int, help='How many cores for multiprocessing? (e.g. 4 uses 4, -2 uses all but 2)')
    # directory settings
    parser.add_argument('--sim_dir',            dest='sim_dir', type=str, help='Directory for raw simulated data')
    parser.add_argument('--fmt_dir',            dest='fmt_dir', type=str, help='Directory for tensor-formatted simulated data')
    parser.add_argument('--net_dir',            dest='net_dir', type=str, help='Directory for trained networks and predictions')
    parser.add_argument('--plt_dir',            dest='plt_dir', type=str, help='Directory for plotted results')
    parser.add_argument('--pred_dir',           dest='pred_dir', type=str, help='Predict results for dataset located in this directory')
    # model settings
    #parser.add_argument('--show_models',        dest='show_models', type=bool, default=False, help='Print all available model types and variants?')
    parser.add_argument('--model_type',         dest='model_type', type=str, help='Model type')
    parser.add_argument('--model_variant',      dest='model_variant', type=str, help='Model variant')
    parser.add_argument('--num_char',            dest='num_char', type=int, help='Number of characters')
    # simulation settings
    parser.add_argument('--sim_logging',        dest='sim_logging', type=str, choices=['clean', 'verbose', 'compress'], help='Simulation logging style')
    parser.add_argument('--start_idx',          dest='start_idx', type=int, help='Start index for simulation')
    parser.add_argument('--end_idx',            dest='end_idx', type=int, help='End index for simulation')
    parser.add_argument('--stop_time',          dest='stop_time', type=float, help='Maximum duration of evolution for each simulation')
    parser.add_argument('--stop_floor_sizes',   dest='stop_floor_sizes', type=int, help='Minimum number of taxa for each simulation')
    parser.add_argument('--stop_ceil_sizes',    dest='stop_ceil_sizes', type=int, help='Maximum number of taxa for each simulation')
    # formatting settings
    parser.add_argument('--tensor_format',      dest='tensor_format', type=str, choices=['hdf5', 'csv'], help='Storage format for simulation tensors')
    parser.add_argument('--tree_type',          dest='tree_type', type=str, choices=['extant', 'serial'], help='Type of tree')
    parser.add_argument('--save_phyenc_csv',    dest='save_phyenc_csv', type=bool, help='Save encoded phylogenetic tensor encoding to csv?')
    # learning settings
    parser.add_argument('--tree_size',          dest='tree_size', type=int, help='Number of taxa in phylogenetic tensor')
    parser.add_argument('--num_epochs',         dest='num_epochs', type=int, help='Number of learning epochs')
    parser.add_argument('--batch_size',         dest='batch_size', type=int, help='Training batch sizes during learning')
    parser.add_argument('--prop_test',          dest='prop_test', type=float, help='Proportion of data used as test examples (demonstrate trained network performance)')
    parser.add_argument('--prop_validation',    dest='prop_validation', type=float, help='Proportion of data used as validation examples (diagnose network overtraining)')
    parser.add_argument('--prop_calibration',   dest='prop_calibration', type=float, help='Proportion of data used as calibration examples (calibrate conformal prediction intervals)')
    parser.add_argument('--alpha_CQRI',         dest='alpha_CQRI', type=float, help='Expected coverage percent for prediction intervals')
    parser.add_argument('--loss',               dest='loss', type=str, help='Loss function used as optimization criterion')
    parser.add_argument('--optimizer',          dest='optimizer', type=str, help='Method used for optimizing neural network')
    # plotting settings
    parser.add_argument('--network_prefix',     dest='network_prefix', type=str, help='Plot results related to this network prefix')
    # prediction settings
    parser.add_argument('--pred_prefix',        dest='pred_prefix', type=str, help='Predict results for this dataset')

    # parse arguments
    args = parser.parse_args()
    
    # overwrite config_fn is argument passed
    if arg_overwrite and args.config_fn != None:
        config_fn = args.config_fn
    config_fn = config_fn.rstrip('.py')
    
    # config from file
    m = importlib.import_module(config_fn)

    def overwrite_defaults(m, args, var):
        x = getattr(args, var)
        if x is not None:
            # if args.force == True:
            #     m.args[var] = x
            # elif var not in m.args:
            #     m.args[var] = x
            m.args[var] = x
        return m
    
    # update arguments from defaults, when provided
    m = overwrite_defaults(m, args, 'proj')
    m = overwrite_defaults(m, args, 'use_parallel')
    m = overwrite_defaults(m, args, 'num_proc')
    m = overwrite_defaults(m, args, 'sim_dir')
    m = overwrite_defaults(m, args, 'fmt_dir')
    m = overwrite_defaults(m, args, 'net_dir')
    m = overwrite_defaults(m, args, 'plt_dir')
    m = overwrite_defaults(m, args, 'pred_dir')
    #m = overwrite_defaults(m, args, 'show_models')
    m = overwrite_defaults(m, args, 'model_type')
    m = overwrite_defaults(m, args, 'model_variant')
    m = overwrite_defaults(m, args, 'num_char')
    m = overwrite_defaults(m, args, 'sim_logging')
    m = overwrite_defaults(m, args, 'start_idx')
    m = overwrite_defaults(m, args, 'end_idx')
    m = overwrite_defaults(m, args, 'stop_time')
    m = overwrite_defaults(m, args, 'stop_floor_sizes')
    m = overwrite_defaults(m, args, 'stop_ceil_sizes')
    m = overwrite_defaults(m, args, 'tree_size')
    m = overwrite_defaults(m, args, 'save_phyenc_csv')
    m = overwrite_defaults(m, args, 'num_epochs')
    m = overwrite_defaults(m, args, 'batch_size')
    m = overwrite_defaults(m, args, 'prop_test')
    m = overwrite_defaults(m, args, 'prop_validation')
    m = overwrite_defaults(m, args, 'prop_calibration')
    m = overwrite_defaults(m, args, 'alpha_CQRI')
    m = overwrite_defaults(m, args, 'loss')
    m = overwrite_defaults(m, args, 'optimizer')
    m = overwrite_defaults(m, args, 'network_prefix')
    m = overwrite_defaults(m, args, 'pred_prefix')

    # return new args
    return m.args

#-----------------------------------------------------------------------------------------------------------------#

###################
# GENERAL HELPERS #
###################

def make_symm(m):
    d = np.diag(m)
    m = np.triu(m)
    m = m + m.T
    np.fill_diagonal(m, d)
    return m

# Chat-GPT function
def sort_binary_vectors(binary_vectors):
    """
    Sorts a list of binary vectors in order of number of "on" bits first, and then left to right in terms of which bits are "on".
    """
    # Define a helper function to count the number of "on" bits in a binary vector
    def count_ones(binary_vector):
        return sum(binary_vector)
    
    # Sort the binary vectors in the list first by number of "on" bits
    sorted_vectors = sorted(binary_vectors, key=count_ones)
    
    # Sort the binary vectors in the list by "on" bits from left to right
    for i in range(len(sorted_vectors)):
        for j in range(i+1, len(sorted_vectors)):
            if count_ones(sorted_vectors[j]) == count_ones(sorted_vectors[i]):
                for k in range(len(sorted_vectors[i])):
                    if sorted_vectors[i][k] != sorted_vectors[j][k]:
                        if sorted_vectors[j][k] > sorted_vectors[i][k]:
                            sorted_vectors[i], sorted_vectors[j] = sorted_vectors[j], sorted_vectors[i]
                        break
                
    return sorted_vectors


# helper functions
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def find_tree_width(num_taxa, max_taxa):
    if num_taxa == 0:
        return 0
    elif num_taxa > max_taxa[-1]:
        return -1
    for i in max_taxa:
        if num_taxa <= i:
            return i
    # should never call this
    raise Exception('error in find_tree_width()', num_taxa, max_taxa)
    return -2

def clean_scientific_notation(s):
    return re.sub( '\.0+E\+0+', '', s)

#-----------------------------------------------------------------------------------------------------------------#


################
# FILE HELPERS #
################

def write_to_file(s, fn):
    f = open(fn, 'w')
    f.write(s)
    f.close()

def read_tree(tre_fn):
    # check that file exists 
    if not os.path.exists(tre_fn):
        raise FileNotFoundError(f'Could not find tree file at {tre_fn}')
    
    phy = None
    for schema in [ 'newick', 'nexus' ]:
        try:
            phy_tmp = dp.Tree.get(path=tre_fn, schema=schema)
        except:
            phy_tmp = None
        else:
             if phy_tmp is not None:
                phy = phy_tmp
    return phy

def make_clean_phyloenc_str(x):
    s = np.array2string(x, separator=',', max_line_width=1e200, threshold=1e200, edgeitems=1e200, precision=10, floatmode='maxprec')
    s = re.sub(r'[\[\]]', '', string=s)
    s = re.sub(r',\n ', '\n', string=s)
    s = s + '\n'
    return s


#-----------------------------------------------------------------------------------------------------------------#

#####################
# FORMAT CONVERTERS #
#####################

def convert_nexus_to_array(dat_fn):
    
    # read file
    f = open(dat_fn, 'r')
    lines = f.readlines()
    f.close()

    # process file
    found_matrix = False
    num_taxa    = 0
    num_char    = 0
    taxon_idx   = 0
    taxon_names = []
    for line in lines:
        # purge whitespace
        line = ' '.join(line.split()).rstrip('\n')
        tok = line.split(' ')
        
        # skip lines with comments
        if tok[0] == '[':
            continue

        # get data dimenions
        if tok[0].upper() == 'DIMENSIONS':
            for x in tok:
                x = x.rstrip(';')
                if 'NTAX' in x.upper():
                    num_taxa = int(x.split('=')[1])
                elif 'NCHAR' in x.upper():
                    num_char = int(x.split('=')[1])
            dat = np.zeros((num_char, num_taxa), dtype='int')

        # entering data matrix
        if tok[0].upper() == 'MATRIX':
            found_matrix = True
            continue

        # process data matrix
        if found_matrix:
            if tok[0] == ';':
                found_matrix = False
                break
            elif len(tok) == 2:
                #print(tok)
                name = tok[0]
                state = tok[1]
                taxon_names.append(name)
                dat[:,taxon_idx] = [ int(z) for z in state ]
                taxon_idx += 1

    # construct data frame
    # rows: char states
    # cols: taxa
    df = pd.DataFrame(dat, columns=taxon_names)
    
    return df


def convert_table_to_array(dat_fn, sep=","):
    
    # read file
    f = open(dat_fn, 'r')
    lines = f.readlines()
    f.close()

    # process file
    num_taxa    = len(lines)
    num_char    = 0
    taxon_idx   = 0
    taxon_names = []
    first_taxon = True

    for line in lines:
        # purge whitespace
        line = ' '.join(line.split()).rstrip('\n')
        tok = line.split(sep)
        
        # get taxon + state
        name = tok[0]
        state = tok[1]

        # construct matrix based on num char
        if first_taxon:
            first_taxon = False
            num_char = len(state)
            dat = np.zeros((num_char, num_taxa), dtype='int')

        # save taxon name, populate array
        taxon_names.append(name)
        dat[:,taxon_idx] = [ int(z) for z in state ]
        taxon_idx += 1

    # construct data frame
    # rows: char states
    # cols: taxa
    df = pd.DataFrame(dat, columns=taxon_names)
    
    return df


# Converts MASTER output into nex
# move to MasterSimulator?
def convert_phy2dat_nex(phy_nex_fn, int2vec):

    # get num regions from size of bit vector
    num_char = len(int2vec[0])

    # get tip names and states from NHX tree
    nex_file = open(phy_nex_fn, 'r')
    nex_str  = nex_file.readlines()[3]
    matches  = re.findall(pattern='([0-9]+)\[\&type="([A-Z]+)",location="([0-9]+)"', string=nex_str)
    num_taxa = len(matches)
    nex_file.close()

    # generate taxon-state data
    #d = {}
    s_state_str = ''
    for i,v in enumerate(matches):
        taxon        = v[0]
        state        = int(v[2])
        vec_str      = ''.join([ str(x) for x in int2vec[state] ])
        #d[ taxon ]   = vec_str
        s_state_str += taxon + '  ' + vec_str + '\n'
    
    # build new nexus string
    s = \
'''#NEXUS
Begin DATA;
Dimensions NTAX={num_taxa} NCHAR={num_char}
Format MISSING=? GAP=- DATATYPE=STANDARD SYMBOLS="01";
Matrix
{s_state_str}
;
END;
'''.format(num_taxa=num_taxa, num_char=num_char, s_state_str=s_state_str)

    return s

## set return None if bad, then flag the index as a bad sim.
def make_prune_phy(tre_fn, prune_fn):
    # read tree
    phy_ = dp.Tree.get(path=tre_fn, schema='newick')
    # compute all root-to-node distances
    root_distances = phy_.calc_node_root_distances()
    # find tree height (max root-to-node distance)
    tree_height = np.max( root_distances )
    # create empty dictionary
    d = {}
    # loop through all leaf nodes
    leaf_nodes = phy_.leaf_nodes()
    for i,nd in enumerate(leaf_nodes):
        # convert root-distances to ages
        age = tree_height - nd.root_distance
        nd.annotations.add_new('age', age)
        # ultrametricize ages for extant taxa
        if age < 1e-6:
            age = 0.0
        # store taxon and age in dictionary
        taxon_name = str(nd.taxon).strip('\'')
        taxon_name = taxon_name.replace(' ', '_')
        d[ taxon_name ] = age
    # determine what to drop
    drop_taxon_labels = [ k for k,v in d.items() if v > 1e-12 ]
    # inform user if pruning yields valid tree
    if len(leaf_nodes) - len(drop_taxon_labels) < 2:
        return False
    else:
        # prune non-extant taxa
        phy_.prune_taxa_with_labels( drop_taxon_labels )
        # write pruned tree
        phy_.write(path=prune_fn, schema='newick')
        return True



# Used in Encoding
def settings_to_str(settings, taxon_category):
    s = 'setting,value\n'
    s += 'model_name,' + settings['model_name'] + '\n'
    s += 'model_type,' + settings['model_type'] + '\n'
    s += 'replicate_index,' + str(settings['replicate_index']) + '\n'
    s += 'taxon_category,' + str(taxon_category) + '\n'
    return s

def param_dict_to_str(params):
    s1 = 'param,i,j,value\n'
    s2 = ''
    s3 = ''
    for k,v in params.items():
        for i,x in enumerate(v):
            if len(v.shape) == 1:
                rate = np.round(x, NUM_DIGITS)
                s1 += '{k},{i},{i},{v}\n'.format(k=k,i=i,v=rate)
                s2 += '{k}_{i},'.format(k=k,i=i)
                s3 += str(rate) + ','
            else:
                for j,y in enumerate(x):
                    rate = np.round(y, NUM_DIGITS)
                    s1 += '{k},{i},{j},{v}\n'.format(k=k,i=i,j=j,v=rate)
                    s2 += '{k}_{i}_{j},'.format(k=k,i=i,j=j)
                    s3 += str(rate) + ','

    s4 = s2.rstrip(',') + '\n' + s3.rstrip(',') + '\n'
    return s1,s4

def events2df(events):
    df = pd.DataFrame({
        'name'     : [ e.name for e in events ],
        'group'    : [ e.group for e in events ], 
        'i'        : [ e.i for e in events ],
        'j'        : [ e.j for e in events ],
        'k'        : [ e.k for e in events ],
        'reaction' : [ e.reaction for e in events ],
        'rate'     : [ e.rate for e in events ]
    })
    return df

def states2df(states):
    df = pd.DataFrame({
        'lbl' : states.int2lbl,
        'int' : states.int2int,
        'set' : states.int2set,
        'vec' : states.int2vec
    })
    return df

#-----------------------------------------------------------------------------------------------------------------#

#########################
# PHYLO TENSOR ENCODING #
#########################

# ==> move to Formatting? <==

def encode_phy_tensor(phy, dat, tree_width, tree_type, rescale=True):
    if tree_type == 'serial':
        phy_tensor = encode_cblvs(phy, dat, tree_width, rescale)
    elif tree_type == 'extant':
        phy_tensor = encode_cdvs(phy, dat, tree_width, rescale)
    else:
        ValueError(f'Unrecognized {tree_type}')
    return phy_tensor

def encode_cdvs(phy, dat, tree_width, rescale=True):
    
    # num columns equals tree_size, 0-padding
    # returns tensor with following rows
    # 0: terminal brlen, 1: last-int-node brlen, 2: last-int-node root-dist
    
    # data dimensions
    num_char  = dat.shape[0]

    # initialize workspace
    root_distances = phy.calc_node_root_distances(return_leaf_distances_only=False)
    heights    = np.zeros( (3, tree_width) )
    states     = np.zeros( (num_char, tree_width) )
    state_idx  = 0
    height_idx = 0

    # postorder traversal to rotate nodes by clade-length
    for nd in phy.postorder_node_iter():
        if nd.is_leaf():
            nd.treelen = 0.
        else:
            children           = nd.child_nodes()
            ch_treelen         = [ (ch.edge.length + ch.treelen) for ch in children ]
            nd.treelen         = sum(ch_treelen)
            ch_treelen_rank    = np.argsort( ch_treelen )[::-1] 
            children_reordered = [ children[i] for i in ch_treelen_rank ]
            nd.set_children(children_reordered)

    # inorder traversal to fill matrix
    phy.seed_node.edge.length = 0
    for nd in phy.inorder_node_iter():
        
        if nd.is_leaf():
            heights[0,height_idx] = nd.edge.length
            states[:,state_idx]   = dat[nd.taxon.label].to_list()
            state_idx += 1
        else:
            heights[1,height_idx] = nd.edge.length
            heights[2,height_idx] = nd.root_distance
            height_idx += 1

    # fill in phylo tensor
    if rescale:
        heights = heights / np.max(heights)
    phylo_tensor = np.vstack( [heights, states] )

    return phylo_tensor


def encode_cblvs(phy, dat, tree_width, rescale=True):
    # data dimensions
    num_char   = dat.shape[0]

    # initialize workspace
    null       = phy.calc_node_root_distances(return_leaf_distances_only=False)
    heights    = np.zeros( (4, tree_width) ) 
    states     = np.zeros( (num_char, tree_width) )
    state_idx  = 0
    height_idx = 0

    # postorder traversal to rotate nodes by max-root-distance
    for nd in phy.postorder_node_iter():
        if nd.is_leaf():
            nd.max_root_distance = nd.root_distance
        else:
            children                  = nd.child_nodes()
            ch_max_root_distance      = [ ch.max_root_distance for ch in children ]
            ch_max_root_distance_rank = np.argsort( ch_max_root_distance )[::-1] # [0,1] or [1,0]
            children_reordered        = [ children[i] for i in ch_max_root_distance_rank ]
            nd.max_root_distance      = max(ch_max_root_distance)
            nd.set_children(children_reordered)

    # inorder traversal to fill matrix
    last_int_node = phy.seed_node
    last_int_node.edge.length = 0
    for nd in phy.inorder_node_iter():
        if nd.is_leaf():
            heights[0,height_idx] = nd.edge.length
            heights[2,height_idx] = nd.root_distance - last_int_node.root_distance
            states[:,state_idx]   = dat[nd.taxon.label].to_list()
            state_idx += 1
        else:
            #print(last_int_node.edge.length)
            heights[1,height_idx+1] = nd.edge.length
            heights[3,height_idx+1] = nd.root_distance
            last_int_node = nd
            height_idx += 1

    # fill in phylo tensor
    #heights.shape = (2, tree_size)
    # 0: leaf brlen; 1: intnode brlen; 2:leaf-to-lastintnode len; 3:lastintnode-to-root len
    if rescale:
        heights = heights / np.max(heights)
    phylo_tensor = np.vstack( [heights, states] )

    return phylo_tensor


#-----------------------------------------------------------------------------------------------------------------#

#################
# SUMMARY STATS #
#################

# ==> move to Formatting? <==

def make_summ_stat(tre_fn, geo_fn, states_bits_str_inv):
    
    # build summary stats
    summ_stats = {}

    # read tree + states
    phy = dp.Tree.get(path=tre_fn, schema="newick")
    num_taxa                  = len(phy.leaf_nodes())
    root_distances            = phy.calc_node_root_distances()
    tree_height               = np.max( root_distances )
    branch_lengths            = [ nd.edge.length for nd in phy.nodes() if nd != phy.seed_node ]

    # tree statistics
    summ_stats['n_taxa']      = num_taxa
    summ_stats['tree_length'] = phy.length()
    summ_stats['tree_height'] = tree_height
    summ_stats['brlen_mean']  = np.mean(branch_lengths)
    summ_stats['brlen_var']   = np.var(branch_lengths)
    #summ_stats['brlen_skew']  = sp.stats.skew(branch_lengths)
    #summ_stats['brlen_kurt']  = sp.stats.kurtosis(branch_lengths)
    summ_stats['age_mean']    = np.mean(root_distances)
    summ_stats['age_var']     = np.var(root_distances)
    #summ_stats['age_skew']    = sp.stats.skew(root_distances)
    #summ_stats['age_kurt']    = sp.stats.kurtosis(root_distances)
    summ_stats['B1']          = dp.calculate.treemeasure.B1(phy)
    summ_stats['N_bar']       = dp.calculate.treemeasure.N_bar(phy)
    summ_stats['colless']     = dp.calculate.treemeasure.colless_tree_imbalance(phy)
    summ_stats['treeness']    = dp.calculate.treemeasure.treeness(phy)
    #summ_stats['gamma']       = dp.calculate.treemeasure.pybus_harvey_gamma(phy)
    #summ_stats['sackin']      = dp.calculate.treemeasure.sackin_index(phy)

    # read characters + states
    f = open(geo_fn, 'r')
    m = f.read().splitlines()
    f.close()
    y = re.search(string=m[2], pattern='NCHAR=([0-9]+)')
    z = re.search(string=m[3], pattern='SYMBOLS="([0-9A-Za-z]+)"')
    num_char = int(y.group(1))
    states = z.group(1)
    #num_states = len(states)
    #num_combo = num_char * num_states

    # get taxon data
    taxon_state_block = m[ m.index('Matrix')+1 : m.index('END;')-1 ]
    taxon_states = [ x.split(' ')[-1] for x in taxon_state_block ]

    # freqs of entire char-set
    # freq_taxon_states = np.zeros(num_char, dtype='float')
    for i in range(num_char):
        summ_stats['n_char_' + str(i)] = 0
        summ_stats['f_char_' + str(i)] = 0.
    for k in list(states_bits_str_inv.keys()):
        #freq_taxon_states[ states_bits_str_inv[k] ] = taxon_states.count(k) / num_taxa
        summ_stats['n_state_' + str(k)] = taxon_states.count(k)
        summ_stats['f_state_' + str(k)] = taxon_states.count(k) / num_taxa
        for i,j in enumerate(k):
            if j != '0':
                summ_stats['n_char_' + str(i)] += summ_stats['n_state_' + k]
                summ_stats['f_char_' + str(i)] += summ_stats['f_state_' + k]

    return summ_stats

def make_summ_stat_str(ss):
    keys_str = ','.join( list(ss.keys()) ) + '\n'
    vals_str = ','.join( [ str(x) for x in ss.values() ] ) + '\n'
    return keys_str + vals_str


#-----------------------------------------------------------------------------------------------------------------#

#########################
# Tensor de/normalizing #
#########################

def normalize(data, m_sd = None):
    if(type(m_sd) == type(None)):
        m = data.mean(axis = 0)
        sd = data.std(axis = 0)
        sd[np.where(sd == 0)] = 1
        return (data - m)/sd, m, sd
    else:
        m_sd[1][np.where(m_sd[1] == 0)] = 1
        return (data - m_sd[0])/m_sd[1]
        
    
def denormalize(data, train_mean, train_sd, log_labels = False):
    return data * train_sd + train_mean

#-----------------------------------------------------------------------------------------------------------------#

#######################
# CQR functions      ##
#######################

# ==> Move to Learning? <==

def pinball_loss(y_true, y_pred, alpha):
    err = y_true - y_pred
    return K.mean(K.maximum(alpha*err, (alpha-1)*err), axis=-1)

def pinball_loss_q_0_025(y_true, y_pred):
    return pinball_loss(y_true, y_pred, alpha=0.025)

def pinball_loss_q_0_975(y_true, y_pred):
    return pinball_loss(y_true, y_pred, alpha=0.975)

def pinball_loss_q_0_05(y_true, y_pred):
    return pinball_loss(y_true, y_pred, alpha=0.05)

def pinball_loss_q_0_95(y_true, y_pred):
    return pinball_loss(y_true, y_pred, alpha=0.95)

def pinball_loss_q_0_10(y_true, y_pred):
    return pinball_loss(y_true, y_pred, alpha=0.10)

def pinball_loss_q_0_90(y_true, y_pred):
    return pinball_loss(y_true, y_pred, alpha=0.90)

def pinball_loss_q_0_15(y_true, y_pred):
    return pinball_loss(y_true, y_pred, alpha=0.15)

def pinball_loss_q_0_85(y_true, y_pred):
    return pinball_loss(y_true, y_pred, alpha=0.85)

# computes the distance y_i is inside/outside the lower(x_i) and upper(x_i) quantiles
# there are three cases to consider:
#   1. y_i is under the lower bound: max-value will be q_lower(x_i) - y_i & positive
#   2. y_i is over the upper bound:  max-value will be y_i - q_upper(x_i) & positive
#   3. y_i is between the bounds:    max-value will be the difference between y_i and the closest bound & negative
def compute_conformity_scores(x, y, q_lower, q_upper):
    return np.max( q_lower(x)-y, y-q_upper(x) )


# def get_CQR_constant(x_pred_quantiles, y_true, inner_quantile=0.95):
#     # preds axis 0 is the lower and upper quants, axis 1 is the replicates, and axis 2 is the param label
#     # compute non-comformity scores
#     Q = np.array([])
#     # error tolerance on quantile for E
#     error = 0.001
#     # for each parameter
#     #inner_quantile * (1 + 1/x_pred_quantiles.shape[1])
#     for i in range(x_pred_quantiles.shape[2]):
#         E = np.amax(np.array((x_pred_quantiles[0][:,i] - y_true[:,i], y_true[:,i] - x_pred_quantiles[1][:,i])), axis=0)

#         # get 1 - alpha/2's quintile of non-comformity scores
#         #print( inner_quantile * (1 + 1/x_pred_quantiles.shape[1]) )
#         quant = inner_quantile * (1 + 1/x_pred_quantiles.shape[1])
#         # if quant < 0 and quant > 0 - error:
#         #     quant = 0.
#         # elif quant > 1. and quant < 1. + error:
#         #     quant = 1.
#         Q = np.append(Q, np.quantile(E, quant))

#     return Q
def get_CQR_constant_old(preds, true, inner_quantile=0.95, symmetric = True):
    #preds axis 0 is the lower and upper quants, axis 1 is the replicates, and axis 2 is the params
    # compute non-comformity scores
    Q = np.array([]) if symmetric else np.empty((2, preds.shape[2]))
    for i in range(preds.shape[2]):
        if symmetric:
            # Symmetric non-comformity score
            s = np.amax(np.array((preds[0][:,i] - true[:,i], true[:,i] - preds[1][:,i])), axis=0)
            # get adjustment constant: 1 - alpha/2's quintile of non-comformity scores
            Q = np.append(Q, np.quantile(s, inner_quantile * (1 + 1/preds.shape[1])))
        else:
            # Asymmetric non-comformity score
            lower_s = np.array(true[:,i] - preds[0][:,i])
            upper_s = np.array(true[:,i] - preds[1][:,i])
            # get (lower_q adjustment, upper_q adjustment)
            Q[:,i] = np.array((np.quantile(lower_s, (1 - inner_quantile)/2 * (1 + 1/preds.shape[1])),
                               np.quantile(upper_s, (1 + inner_quantile)/2 * (1 + 1/preds.shape[1]))))
    return Q

def get_CQR_constant(preds, true, inner_quantile=0.95, symmetric = True):
    #preds axis 0 is the lower and upper quants, axis 1 is the replicates, and axis 2 is the params
    # compute non-comformity scores
    Q = np.empty((2, preds.shape[2]))
    
    for i in range(preds.shape[2]):
        if symmetric:
            # Symmetric non-comformity score
            s = np.amax(np.array((preds[0][:,i] - true[:,i], true[:,i] - preds[1][:,i])), axis=0)
            # get adjustment constant: 1 - alpha/2's quintile of non-comformity scores
            #Q = np.append(Q, np.quantile(s, inner_quantile * (1 + 1/preds.shape[1])))
            lower_q = np.quantile(s, inner_quantile * (1 + 1/preds.shape[1]))
            upper_q = lower_q
            #Q[:,i] = np.array([lower_q, upper_q])
        else:
            # Asymmetric non-comformity score
            lower_s = np.array(true[:,i] - preds[0][:,i])
            upper_s = np.array(true[:,i] - preds[1][:,i])
            lower_q = np.quantile(lower_s, (1 - inner_quantile)/2 * (1 + 1/preds.shape[1]))
            upper_q = np.quantile(upper_s, (1 + inner_quantile)/2 * (1 + 1/preds.shape[1]))
            # get (lower_q adjustment, upper_q adjustment)

        Q[:,i] = np.array([lower_q, upper_q])
                               
    return Q

# make matrix with parameter values, lower-bounds, upper-bounds: 3D->2D
def make_param_VLU_mtx(A, param_names):
    
    # axis labels
    stat_names = ['value', 'lower', 'upper']

    # multiindex
    index = pd.MultiIndex.from_product([range(s) for s in A.shape], names=['stat', 'rep_idx', 'param'])
    
    # flattened data frame
    df = pd.DataFrame({'A': A.flatten()}, index=index)['A']
    df = df.reorder_levels(['param','stat','rep_idx']).sort_index()

    # unstack stat and param, so they become combined header indices
    df = df.unstack(level=['stat','param'])
    #col_names = df.columns

    new_col_names = [ f'{param_names[y]}_{stat_names[x]}' for x,y in df.columns ]
    df.columns = new_col_names

    return df



#-----------------------------------------------------------------------------------------------------------------#



# graveyard




# def read_tree2(newick_tree):
#     """ Tries all nwk formats and returns an ete3 Tree

#     :param newick_tree: str, a tree in newick format
#     :return: ete3.Tree
#     """
#     tree = None
#     for f in (3, 2, 5, 0, 1, 4, 6, 7, 8, 9):
#         try:
#             tree = Tree(newick_tree, format=f)
#             break
#         except:
#             continue
#     if not tree:
#         raise ValueError('Could not read the tree {}. Is it a valid newick?'.format(newick_tree))
#     return tree


# def read_tree_file(tree_path):
#     with open(tree_path, 'r') as f:
#         nwk = f.read().replace('\n', '').split(';')
#         if nwk[-1] == '':
#             nwk = nwk[:-1]
#     if not nwk:
#         raise ValueError('Could not find any trees (in newick format) in the file {}.'.format(tree_path))
#     if len(nwk) > 1:
#         raise ValueError('There are more than 1 tree in the file {}. Now, we accept only one tree per inference.'.format(tree_path))
#     return read_tree(nwk[0] + ';')


# def check_tree_size(tre):
#     """
#     Verifies whether input tree is of correct size and determines the tree size range for models to use
#     :param tre: ete3.Tree
#     :return: int, tree_size
#     """
#     if 49 < len(tre) < 200:
#         tre_size = 'SMALL'
#     elif 199 < len(tre) < 501:
#         tre_size = 'LARGE'
#     else:
#         raise ValueError('Your input tree is of incorrect size (either smaller than 50 tips or larger than 500 tips.')

#     return tre_size


# def annotator(predict, mod):
#     """
#     annotates the pd.DataFrame containing predicted values
#     :param predict: predicted values
#     :type: pd.DataFrame
#     :param mod: model under which the parameters were estimated
#     :type: str
#     :return:
#     """

#     if mod == "BD":
#         predict.columns = ["R_naught", "Infectious_period"]
#     elif mod == "BDEI":
#         predict.columns = ["R_naught", "Infectious_period", "Incubation_period"]
#     elif mod == "BDSS":
#         predict.columns = ["R_naught", "Infectious_period", "X_transmission", "Superspreading_individuals_fraction"]
#     elif mod == "BD_vs_BDEI_vs_BDSS":
#         predict.columns = ["Probability_BDEI", "Probability_BD", "Probability_BDSS"]
#     elif mod == "BD_vs_BDEI":
#         predict.columns = ["Probability_BD", "Probability_BDEI"]
#     return predict


# def rescaler(predict, rescale_f):
#     """
#     rescales the predictions back to the initial tree scale (e.g. days, weeks, years)
#     :param predict: predicted values
#     :type: pd.DataFrame
#     :param rescale_f: rescale factor by which the initial tree was scaled
#     :type: float
#     :return:
#     """

#     for elt in predict.columns:
#         if "period" in elt:
#             predict[elt] = predict[elt]*rescale_f

#     return predict



# #####################
# # FORMAT CONVERSION #
# #####################

# def make_cblvs_geosse(cblv_df, taxon_states, new_order):
    
#     # array dimensions for GeoSSE states
#     n_taxon_cols = cblv_df.shape[1]
#     n_region = len(list(taxon_states.values())[0])

#     # create states array
#     states_df = np.zeros( shape=(n_region,n_taxon_cols), dtype='int')
    
#     # populate states (not sure if this is how new_order works!)
#     for i,v in enumerate(new_order):
#         y =  [ int(x) for x in taxon_states[v] ]
#         z = np.array(y, dtype='int' )
#         states_df[:,i] = z

#     # append states
#     cblvs_df = np.concatenate( (cblv_df, states_df), axis=0 )
#     cblvs_df = cblvs_df.flatten() #T.reshape((1,-1))
#     #cblvs_df = cblvs_df.T.reshape((1,-1))
#     #cblvs_df.shape = (1,-1)

#     # done!
#     return cblvs_df

# def make_cdv_geosse(cdv_df, taxon_states, new_order):
    
#     # array dimensions for GeoSSE states
#     n_taxon_cols = cdv_df.shape[1]
#     n_region = len(list(taxon_states.values())[0])

#     # create states array
#     states_df = np.zeros( shape=(n_region,n_taxon_cols), dtype='int')
    
#     # populate states (not sure if this is how new_order works!)
#     for i,v in enumerate(new_order):
#         y =  [ int(x) for x in taxon_states[v] ]
#         z = np.array(y, dtype='int' )
#         states_df[:,i] = z

#     # append states
#     cblvs_df = np.concatenate( (cblv_df, states_df), axis=0 )
#     cblvs_df = cblvs_df.T.reshape((1,-1))
#     #cblvs_df.shape = (1,-1)

#     # done!
#     return cblvs_df


#####################################
# DATA DIMENSION/VALIDATION HELPERS #
#####################################

# def get_num_taxa(tre_fn):
#     try:
#         tree = read_tree_file(tre_fn)
#         num_taxa = len(tree.get_leaves())
#     except ValueError:
#         return 0
#     return num_taxa



#############
# 1. translate master output into char matrix
# 2. determine size class of tree
# 3. call make_phy_tensor(phy, dat, tree_size, tree_type) function
# 4. make_phy_tensor calls make_cblvs or make_cdvs, depending on input
# 5. return numpy array

#-----------------------------------------------------------------------------------------------------------------#

################
# CDVS ENCODER #
################

# ### this is where state info is stored
# def set_attribs(tre):
#     """
#     adds t_s attributes to tips based on tip name
#     :param tre: ete3.Tree, the tree on which we measure the branch length
#     :return: void, returns modified tree
#     """
#     for tip in tre.traverse():
#         if "&&NHX-t_s=1" in tip.name:
#             setattr(tip, T_S, 1)
#         elif "&&NHX-t_s=2" in tip.name:
#             setattr(tip, T_S, 2)
#     return None

# def attach_tip_states(tr, st):
#     "assign states in st to leaf nodes in tr using shared taxon name"
#     for nd in tr.get_leaves():
#         setattr(nd, STATE, st[nd.name])

# def get_average_branch_length(tre):
#     """
#     Returns average branch length for given tree
#     :param tre: ete3.Tree, the tree on which we measure the branch length
#     :return: float, average branch length
#     """
#     br_length = [nod.dist for nod in tre.traverse()]
#     return np.average(br_length)


# def rescale_tree(tr, rescale_fac):
#     """
#     Rescales a given tree
#     :param tr: ete3.Tree, the tree to be rescaled
#     :param rescale_fac: float, the branches will be multiplied by this factor
#     :return: void, modifies the original tree
#     """
#     for node in tr.traverse():
#         node.dist = node.dist/rescale_fac
#     return None


# def add_diversification(tr):
#     """
#     to each node adds an attribute, 'diversification_score', i.e. the sum of pathways of branched tips
#     :param tr: ete3.Tree, the tree to be modified
#     :return: void, modifies the original tree
#     """
#     for node in tr.traverse("postorder"):
#         if not node.is_root():
#             # print(label_count)
#             label_node = 0
#             if node.is_leaf():
#                 label_node = 1
#                 setattr(node, DIVERSIFICATION_SCORE, node.dist)
#             else:
#                 children = node.get_children()
#                 # print(children)
#                 setattr(node, DIVERSIFICATION_SCORE, getattr(children[0], DIVERSIFICATION_SCORE) + getattr(children[1], DIVERSIFICATION_SCORE))
#     return None


# def add_diversification_sign(tr):
#     """
#     Puts topological signatures based on diversification (i.e. longest path): if the first child of a node has longer
#     path of branches leading to it, then it is prioritized for visit.
#     :param tr: ete3.Tree, the tree to get the topological description
#     :return: void, modifies the original tree
#     """
#     for node in tr.traverse('levelorder'):
#         if not node.is_leaf():
#             diver_child0 = getattr(node.children[0], DIVERSIFICATION_SCORE)
#             diver_child1 = getattr(node.children[1], DIVERSIFICATION_SCORE)
#             if diver_child0 < diver_child1:
#                 node.add_feature(TURN_ONE, True)
#             elif diver_child0 == diver_child1:
#                 next_sign = random.choice([True, False])
#                 if next_sign is True:
#                     node.add_feature(TURN_ONE, True)
#             else:
#                 node.add_feature(TURN_ONE, False)
#     return None


# def name_tree_cdvs(tr):
#     """
#     Names all the tree nodes that are not named, with unique names.
#     :param tr: ete3.Tree, the tree to be named
#     :return: void, modifies the original tree
#     """
#     i = 0
#     for node in tr.traverse('levelorder'):
#         node.name = i
#         i += 1
#     return None


# def add_dist_to_root_cdvs(tr):
#     # int_nodes_dist = []
#     # tips_dist = []
#     tree_height = 0
#     for node in tr.traverse("preorder"):
#         if node.is_root():
#             node.add_feature("dist_to_root", 0)
#         elif node.is_leaf():
#             node.add_feature("dist_to_root", getattr(node.up, "dist_to_root") + node.dist)
#             # tips_dist.append(getattr(node.up, "dist_to_root") + node.dist)
#             tree_height = getattr(node, "dist_to_root", False)

#         else:
#             node.add_feature("dist_to_root", getattr(node.up, "dist_to_root") + node.dist)
#             # int_nodes_dist.append(getattr(node.up, "dist_to_root") + node.dist)
#     return tr, tree_height


# def get_not_visited_anc(leaf):
#     while getattr(leaf, "visited", False):
#         leaf = leaf.up
#     return leaf


# def get_dist_to_root(anc):
#     dist_to_root = getattr(anc, "dist_to_root")
#     return dist_to_root


# def follow_signs(anc):
#     end_leaf = anc
#     while not end_leaf.is_leaf():
#         if getattr(end_leaf, TURN_ONE, False):
#             if getattr(end_leaf.children[1], 'visited', False):
#                 end_leaf = end_leaf.children[0]
#             else:
#                 end_leaf = end_leaf.children[1]
#         else:
#             if getattr(end_leaf.children[0], 'visited', False):
#                 end_leaf = end_leaf.children[1]
#             else:
#                 end_leaf = end_leaf.children[0]
#     return end_leaf


# def enc_diver(anc):
#     leaf = follow_signs(anc)
#     #print(leaf.state)
#     #print([ int(x) for x in leaf.state] )
#     #s = [ int(x) for x in leaf.state ]
#     yield str(leaf.state)
#     #for s in [ int(x) for x in leaf.state ]:
#     #    yield s
#     #yield float(leaf.state)
#     setattr(leaf, 'visited', True)
#     anc = get_not_visited_anc(leaf)
#     if anc is None:
#         # print("what")
#         return
#     setattr(anc, 'visited', True)
#     yield get_dist_to_root(anc)
#     for _ in enc_diver(anc):
#         yield _


# def type_count(tr, st, lbl):
#     """
#     Returns the counts of type1 and type2 tips
#     :param tr: ete3.Tree, the tree to be named
#     :return: tuple, counts of type 1 and type 2
#     """
#     counts = dict.fromkeys(lbl, 0)
#     #t1 = 0
#     #t2 = 0
#     for leaf in tr:
#         counts[ leaf.state ] = counts[ leaf.state ] + 1
    
#     #print(counts)
#     return list(counts.values())


# def complete_coding_old(encoding, max_length):
#     add_vect = np.repeat(0, max_length - len(encoding))
#     add_vect = list(add_vect)
#     encoding.extend(add_vect)
#     return encoding

# def complete_coding(encoding, max_length):
#     #add_vect = np.repeat(0, max_length - len(encoding))
#     num_row,num_col = encoding.shape
#     add_zeros = np.zeros( (num_row, max_length-num_col) )
#     #add_vect = np.repeat(0, max_length - num_col)
#     #encoding = np.append(encoding, add_vect)
#     #encoding.extend(add_vect)
#     encoding = np.hstack( [encoding, add_zeros] )
#     return encoding

# def expand_tip_states(tips_info):
#     n_idx = len(tips_info)
#     n_char = len(tips_info[1])
#     #print(n_char, n_idx)
#     x = np.zeros( shape=(n_char, n_idx) )
#     #x[:,0] = tips_info[0]  # needed?
#     for i in range(n_idx):
#         x[:,i] = [ int(y) for y in tips_info[i] ]
#     return x


# def make_cdvs(tree_fn, max_len, states, state_labels):
    
#     # read tree
#     file = open(tree_fn, mode="r")
#     tree_str = file.read()
#     tree = Tree(tree_str, format=1)
    
#     # assign states to tips
#     attach_tip_states(tree, states)
#     name_tree_cdvs(tree)

#     # rescale tree to average branch length of 1
#     # measure average branch length
#     rescale_factor = get_average_branch_length(tree)

#     # rescale tree
#     rescale_tree(tree, rescale_factor)

#     # add dist to root attribute
#     tree, tr_height = add_dist_to_root_cdvs(tree)

#     # add pathway of visiting priorities for encoding
#     add_diversification(tree)
#     add_diversification_sign(tree)

#     # encode the tree
#     tree_embedding = list(enc_diver(tree))
    
#     # separate info on tips and nodes:
#     tips_info = [tree_embedding[i] for i in range(len(tree_embedding)) if i % 2 == 0]
#     node_info = [tree_embedding[i] for i in range(len(tree_embedding)) if i % 2 == 1]
#     node_info.insert(0,0) # pad with zero to align length of info vec ??

#     # expand tip states
#     tips_info = expand_tip_states(tips_info)
#     node_info = np.array([node_info])

#     # complete embedding
#     tips_info = complete_coding(tips_info, max_len)
#     node_info = complete_coding(node_info, max_len)

#     # vertical stack
#     complete_info = np.vstack( [node_info, tips_info] )
    
#     # flatten
#     complete_info = complete_info.flatten()
    
#     return complete_info
    
#-----------------------------------------------------------------------------------------------------------------#

#################
# CBLVS ENCODER #
#################


# def add_dist_to_root(tre):
#     """
#     Add distance to root (dist_to_root) attribute to each node
#     :param tre: ete3.Tree, tree on which the dist_to_root should be added
#     :return: void, modifies the original tree
#     """

#     for node in tre.traverse("preorder"):
#         if node.is_root():
#             node.add_feature("dist_to_root", 0)
#         elif node.is_leaf():
#             node.add_feature("dist_to_root", getattr(node.up, "dist_to_root") + node.dist)
#             # tips_dist.append(getattr(node.up, "dist_to_root") + node.dist)
#         else:
#             node.add_feature("dist_to_root", getattr(node.up, "dist_to_root") + node.dist)
#             # int_nodes_dist.append(getattr(node.up, "dist_to_root") + node.dist)
#     return None


# def name_tree(tre, newLeafKeys_inputNameValues):
#     """
#     Names all the tree nodes that are not named, with unique names.
#     :param tre: ete3.Tree, the tree to be named
#     :return: void, modifies the original tree
#     """
#     existing_names = Counter((_.name for _ in tre.traverse() if _.name))
    
#     i = 0
#     for node in tre.traverse('levelorder'):
#         if(node.is_leaf()): # A.M.T
#         	#new_leaf_order_names.append((i, node.name))
#                 newLeafKeys_inputNameValues[i] = node.name
#         node.name = i
#         i += 1
   
#     return None


# def rescale_tree(tre, target_avg_length):
#     """
#     Returns branch length metrics (all branches taken into account and external only)
#     :param tre: ete3.Tree, tree on which these metrics are computed
#     :param target_avg_length: float, the average branch length to which we want to rescale the tree
#     :return: float, resc_factor
#     """
#     # branch lengths
#     dist_all = [node.dist for node in tre.traverse("levelorder")]

#     all_bl_mean = np.mean(dist_all)

#     resc_factor = all_bl_mean/target_avg_length

#     for node in tre.traverse():
#         node.dist = node.dist/resc_factor

#     return resc_factor

    


# def encode_into_most_recent(tree_input, max_taxa=[500], summ_stat=[], target_average_brlen=1.0):
#     """Rescales all trees from tree_file so that mean branch length is 1,
#     then encodes them into full tree representation (most recent version)

#     :param tree_input: ete3.Tree, that we will represent in the form of a vector
#     :param sampling_proba: float, value between 0 and 1, presumed sampling probability value
#     :return: pd.Dataframe, encoded rescaled input trees in the form of most recent, last column being
#      the rescale factor
#     """
#     leaf_ordered_names = [] # A.M.T
#     new_leaf_order_names = []
#     newLeafKeys_inputNameValues = {}

#     # do we want nested functions like this???
#     def real_polytomies(tre):
#         """
#         Replaces internal nodes of zero length with real polytomies.
#         :param tre: ete3.Tree, the tree to be modified
#         :return: void, modifies the original tree
#         """
#         for nod in tre.traverse("postorder"):
#             if not nod.is_leaf() and not nod.is_root():
#                 if nod.dist == 0:
#                     for child in nod.children:
#                         nod.up.add_child(child)
#                     nod.up.remove_child(nod)
#         return

#     def get_not_visited_anc(leaf):
#         while getattr(leaf, "visited", 0) >= len(leaf.children)-1:
#             leaf = leaf.up
#             if leaf is None:
#                 break
#         return leaf

#     def get_deepest_not_visited_tip(anc):
#         max_dist = -1
#         tip = None
#         for leaf in anc:
#             if leaf.visited == 0:
#                 distance_leaf = getattr(leaf, "dist_to_root") - getattr(anc, "dist_to_root")
#                 if distance_leaf > max_dist:
#                     max_dist = distance_leaf
#                     tip = leaf
#         leaf_ordered_names.append(getattr(tip, "name")) # A.M.T
#         return tip

#     def get_dist_to_root(anc):
#         dist_to_root = getattr(anc, "dist_to_root")
#         return dist_to_root

#     def get_dist_to_anc(feuille, anc):
#         dist_to_anc = getattr(feuille, "dist_to_root") - getattr(anc, "dist_to_root")
#         return dist_to_anc

#     def encode(anc):
#         leaf = get_deepest_not_visited_tip(anc)
#         new_leaf_order_names.append(leaf.name) # A.M.T.
#         yield get_dist_to_anc(leaf, anc)
#         leaf.visited += 1
#         anc = get_not_visited_anc(leaf)

#         if anc is None:
#             return
#         anc.visited += 1
#         yield get_dist_to_root(anc)
#         for _ in encode(anc):
#             yield _

#     def complete_coding(encoding, cblv_length):
#         #print(encoding, max_length, max_length - len(encoding) )
#         add_vect = np.repeat(0, cblv_length - len(encoding))
#         add_vect = list(add_vect)
#         encoding.extend(add_vect)
#         return encoding

#     def refactor_to_final_shape(result_v, maxl, summ_stat=[]):
#         def reshape_coor(max_length):
#             tips_coor = np.arange(0, max_length, 2)  # second row
#             #tips_coor = np.insert(tips_coor, -1, max_length + 1)
#             int_nodes_coor = np.arange(1, max_length - 1, 2) # first row
#             int_nodes_coor = np.insert(int_nodes_coor, 0, max_length) # prepend 0??
#             #int_nodes_coor = np.insert(int_nodes_coor, -1, max_length + 2)
#             order_coor = np.append(int_nodes_coor, tips_coor)
#             return order_coor
       
#         #print('test')
#         reshape_coordinates = reshape_coor(maxl)

#         #print(reshape_coordinates.shape)
#         result_v.loc[:, maxl] = 0

#         # reorder the columns        
#         result_v = result_v.iloc[:,reshape_coordinates]

#         return result_v

#     # local copy of input tree
#     tree = tree_input.copy()
    
#     # CBLV size
#     cblv_length = 2*max_taxa
    
#     # remove the edge above root if there is one
#     if len(tree.children) < 2:
#         tree = tree.children[0]
#         tree.detach()

#     # set to real polytomy
#     real_polytomies(tree)

#     # rescale branch lengths
#     rescale_factor = rescale_tree(tree, target_avg_length=target_average_brlen)

#     # set all nodes to non visited:
#     for node in tree.traverse():
#         setattr(node, "visited", 0)

#     name_tree(tree, newLeafKeys_inputNameValues)
    
#     add_dist_to_root(tree)

#     tree_embedding = list(encode(tree))
#     tree_embedding = complete_coding(tree_embedding, cblv_length)
#     result = pd.DataFrame(tree_embedding, columns=[0])
#     result = result.T
#     result = refactor_to_final_shape(result, cblv_length)

#     return result, rescale_factor, new_leaf_order_names, newLeafKeys_inputNameValues

#-----------------------------------------------------------------------------------------------------------------#


# def vectorize_tree_cdv(tre_fn, max_taxa=[500], summ_stat=[], prob=1.0):
#     # get tree and tip labels
#     tree = read_tree_file(tre_fn)    
#     ordered_tip_names = []
#     for i in tree.get_leaves():
#         ordered_tip_names.append(i.name)

#     # returns result, rescale_factor, new_leaf_order_names, newLeafKeys_inputNameValues
#     vv = encode_into_most_recent(tree, max_taxa=max_taxa, summ_stat=summ_stat, target_average_brlen=1.0)
#     otn = np.asarray(ordered_tip_names) # ordered list of the input tip labels
#     vv2 = np.asarray(vv[2]) # ordered list of the new tip labels
#     new_order = [vv[3][i] for i in vv2]

#     if False:
#         print( 'otn ==> ', otn, '\n' )
#         print( 'vv[0] ==>', vv[0], '\n' )
#         print( 'vv[1] ==>', vv[1], '\n' )
#         print( 'vv[2] ==>', vv[2], '\n' )
#         print( 'vv[3] ==>', vv[3], '\n' )

#     cblv = np.asarray( vv[0] )
#     cblv.shape = (2, -1)
#     cblv_df = pd.DataFrame( cblv )

#     return cblv_df,new_order

# def vectorize_tree(tre_fn, max_taxa=500, summ_stat=[], prob=1.0):

#     # get tree and tip labels
#     tree = read_tree_file(tre_fn)    
#     ordered_tip_names = []
#     for i in tree.get_leaves():
#         ordered_tip_names.append(i.name)

#     # returns result, rescale_factor, new_leaf_order_names, newLeafKeys_inputNameValues
#     vv = encode_into_most_recent(tree, max_taxa=max_taxa, summ_stat=summ_stat, target_average_brlen=1.0)
#     otn = np.asarray(ordered_tip_names) # ordered list of the input tip labels
#     vv2 = np.asarray(vv[2]) # ordered list of the new tip labels
#     new_order = [vv[3][i] for i in vv2]

#     #if False:
#     #    print( 'otn ==> ', otn, '\n' )
#     #    print( 'vv[0] ==>', vv[0], '\n' )
#     #    print( 'vv[1] ==>', vv[1], '\n' )
#     #    print( 'vv[2] ==>', vv[2], '\n' )
#     #    print( 'vv[3] ==>', vv[3], '\n' )

#     cblv = np.asarray( vv[0] )
#     cblv.shape = (2, -1)
#     cblv_df = pd.DataFrame( cblv )

#     return cblv_df,new_order


# # Converts MASTER output into nex
# def convert_nex(nex_fn, tre_fn, int2vec):

#     # get num regions from size of bit vector
#     num_char = len(int2vec[0])

#     # get tip names and states from NHX tree
#     nex_file = open(nex_fn, 'r')
#     nex_str = nex_file.readlines()[3]
#     m = re.findall(pattern='([0-9]+)\[\&type="([A-Z]+)",location="([0-9]+)"', string=nex_str)
#     num_taxa = len(m)
#     nex_file.close()

#     # generate taxon-state data
#     d = {}
#     s_state_str = ''
#     for i,v in enumerate(m):
#         taxon = v[0]
#         state = int(v[2])
#         vec_str = ''.join([ str(x) for x in int2vec[state] ])
#         s_state_str += taxon + '  ' + vec_str + '\n'
#         d[ taxon ] = vec_str
    
#     # get newick string (no annotations)
#     tre_file = open(tre_fn, 'r')
#     tre_str = tre_file.readlines()[0]
#     tre_file.close()

#     # build new nexus string
#     s = \
# '''#NEXUS
# Begin DATA;
# Dimensions NTAX={num_taxa} NCHAR={num_char}
# Format MISSING=? GAP=- DATATYPE=STANDARD SYMBOLS="01";
# Matrix
# {s_state_str}
# ;
# END;

# Begin trees;
#     tree 1={tre_str}
# END;
# '''.format(num_taxa=num_taxa, num_char=num_char, tre_str=tre_str, s_state_str=s_state_str)

#     return d,s