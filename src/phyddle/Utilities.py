#!/usr/bin/env python
"""
Utilities
===========
Miscellaneous helper functions phyddle uses for pipeline steps.

Author:    Michael Landis
Copyright: (c) 2023, Michael Landis
License:   MIT
"""

# standard packages
import argparse
import importlib
import re
import os
import sys
import copy
from typing import Optional, List
from itertools import chain, combinations

# external packages
import pandas as pd
import numpy as np
import dendropy as dp


# Precision settings
NUM_DIGITS = 10
np.set_printoptions(floatmode='maxprec', precision=NUM_DIGITS)
pd.set_option('display.precision', NUM_DIGITS)
pd.set_option('display.float_format', lambda x: f'{x:,.3f}')

# Tensorflow info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

##################
# Helper Classes #
##################

# model events
class Event:

    # initialize
    def __init__(self, idx, r=0.0, n=None, g=None, ix=None, jx=None) -> None:
        """
        Creates an event in the model.

        Args:
            idx (dict): A dictionary containing the indices of the event.
            r (float): The rate of the event.
            n (str): The name of the event.
            g (str): The reaction group of the event.
            ix (list): The reaction quantities (reactants) before the event.
            jx (list): The reaction quantities (products) after the event.
        """
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
        return
        
    # make print string
    def make_str(self) -> str:
        """
        Creates a string representation of the event.

        Returns:
            str: The string representation of the event.
        """
        s = 'Event({name},{group},{rate},{idx})'.format(name=self.name, group=self.group, rate=self.rate, idx=self.idx)        
        #s += ')'
        return s
    
    # representation string
    def __repr__(self) -> str:
        """
        Returns the representation of the event.

        Returns:
            str: The representation of the event.
        """
        return self.make_str()
    
    # print string
    def __str__(self) -> str:
        """
        Returns the string representation of the event.

        Returns:
            str: The string representation of the event.
        """
        return self.make_str()


# state space
class States:
    def __init__(self, lbl2vec) -> None:

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
       
        # done
        return

    def make_str(self) -> str:
        """
        Creates a string representation of the state space.

        Returns:
            str: The string representation of the state space.
        """
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
    def __repr__(self) -> str:
        """
        Returns the representation of the state space.

        Returns:
            str: The representation of the state space.
        """
        return self.make_str()
    # print string
    def __str__(self) -> str:
        """
        Returns the string representation of the state space.

        Returns:
            str: The string representation of the state space.
        """
        return self.make_str()
    
    # def make_df(self):
    #     """
    #     ## Probably can delete????
    #     Creates a DataFrame representation of the state space.

    #     Returns:
    #         pandas.DataFrame: The DataFrame representation of the state space.
    #     """
    #     df = pd.DataFrame()
    #     return df



#-----------------------------------------------------------------------------------------------------------------#

###################
# CONFIG LOADER   #
###################

def load_config(config_fn: str,
                arg_overwrite: Optional[bool]=True):
    """
    Loads the configuration.

    Args:
        config_fn (str): The config file name.
        arg_overwrite (bool, optional): Whether to overwrite config file settings with arguments. Defaults to True.

    Returns:
        dict: The loaded configuration.
    """
    
    # argument parsing
    parser = argparse.ArgumentParser(description='phyddle pipeline config') #,
                                     #formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-c', '--cfg',          dest='config_fn', type=str, help='Config file name', metavar='')
    #parser.add_argument('-f', '--force',        action='store_true', help='Arguments override config file settings')
    parser.add_argument('-p', '--proj',         dest='proj', type=str, help='Project name used as directory across pipeline stages', metavar='')
    parser.add_argument('--use_parallel',       dest='use_parallel', type=bool, help='Use parallelization? (recommended)', metavar='')
    parser.add_argument('--num_proc',           dest='num_proc', type=int, help='How many cores for multiprocessing? (e.g. 4 uses 4, -2 uses all but 2)', metavar='')
    # directory settings
    parser.add_argument('--sim_dir',            dest='sim_dir', type=str, help='Directory for raw simulated data', metavar='')
    parser.add_argument('--fmt_dir',            dest='fmt_dir', type=str, help='Directory for tensor-formatted simulated data', metavar='')
    parser.add_argument('--net_dir',            dest='net_dir', type=str, help='Directory for trained networks and predictions', metavar='')
    parser.add_argument('--plt_dir',            dest='plt_dir', type=str, help='Directory for plotted results', metavar='')
    parser.add_argument('--pred_dir',           dest='pred_dir', type=str, help='Predict results for dataset located in this directory', metavar='')
    parser.add_argument('--pred_prefix',        dest='pred_prefix', type=str, help='Predict results for this dataset', metavar='')
    # model settings
    parser.add_argument('--show_models',        action='store_true', help='Print all available model types and variants?')
    parser.add_argument('--model_type',         dest='model_type', type=str, help='Model type', metavar='')
    parser.add_argument('--model_variant',      dest='model_variant', type=str, help='Model variant', metavar='')
    parser.add_argument('--num_char',           dest='num_char', type=int, help='Number of characters', metavar='')
    # simulation settings
    parser.add_argument('--sim_logging',        dest='sim_logging', type=str, choices=['clean', 'verbose', 'compress'], help='Simulation logging style', metavar='')
    parser.add_argument('--start_idx',          dest='start_idx', type=int, help='Start index for simulation', metavar='')
    parser.add_argument('--end_idx',            dest='end_idx', type=int, help='End index for simulation', metavar='')
    parser.add_argument('--stop_time',          dest='stop_time', type=float, help='Maximum duration of evolution for each simulation', metavar='')
    parser.add_argument('--min_num_taxa',       dest='min_num_taxa', type=int, help='Minimum number of taxa for each simulation', metavar='')
    parser.add_argument('--max_num_taxa',       dest='max_num_taxa', type=int, help='Maximum number of taxa for each simulation', metavar='')
    # formatting settings
    parser.add_argument('--tree_type',          dest='tree_type', type=str, choices=['extant', 'serial'], help='Type of tree', metavar='')
    parser.add_argument('--tree_width_cats',    dest='tree_width_cats', type=int, help='The phylo-state tensor widths for formatting training datasets, space-delimited', metavar='')
    parser.add_argument('--tensor_format',      dest='tensor_format', type=str, choices=['hdf5', 'csv'], help='Storage format for simulation tensors', metavar='')
    parser.add_argument('--save_phyenc_csv',    dest='save_phyenc_csv', type=bool, help='Save encoded phylogenetic tensor encoding to csv?', metavar='')
    # learning settings
    parser.add_argument('--tree_width',         dest='tree_width', type=int, help='The phylo-state tensor width dataset used for a neural network', metavar='')
    parser.add_argument('--num_epochs',         dest='num_epochs', type=int, help='Number of learning epochs', metavar='')
    parser.add_argument('--batch_size',         dest='batch_size', type=int, help='Training batch sizes during learning', metavar='')
    parser.add_argument('--prop_test',          dest='prop_test', type=float, help='Proportion of data used as test examples (demonstrate trained network performance)', metavar='')
    parser.add_argument('--prop_validation',    dest='prop_validation', type=float, help='Proportion of data used as validation examples (diagnose network overtraining)', metavar='')
    parser.add_argument('--prop_calibration',   dest='prop_calibration', type=float, help='Proportion of data used as calibration examples (calibrate conformal prediction intervals)', metavar='')
    parser.add_argument('--cpi_coverage',       dest='cpi_coverage', type=float, help='Expected coverage percent for calibrated prediction intervals', metavar='')
    parser.add_argument('--loss',               dest='loss', type=str, help='Loss function used as optimization criterion', metavar='')
    parser.add_argument('--optimizer',          dest='optimizer', type=str, help='Method used for optimizing neural network', metavar='')
    # prediction settings
    # ... nothing for now??
    # plotting settings
    parser.add_argument('--plot_train_color',   dest='plot_train_color', type=str, help='Plotting color for training data elements', metavar='')
    parser.add_argument('--plot_label_color',   dest='plot_label_color', type=str, help='Plotting color for training label elements', metavar='')
    parser.add_argument('--plot_test_color',    dest='plot_test_color', type=str, help='Plotting color for test data elements', metavar='')
    parser.add_argument('--plot_val_color',     dest='plot_val_color', type=str, help='Plotting color for validation data elements', metavar='')
    parser.add_argument('--plot_aux_color',     dest='plot_aux_color', type=str, help='Plotting color for auxiliary input data elements', metavar='')
    parser.add_argument('--plot_pred_color',    dest='plot_pred_color', type=str, help='Plotting color for prediction data elements', metavar='')


    # parse arguments
    args = parser.parse_args()
    
    # print models & exit
    if args.show_models:
         import ModelLoader
         model_str = ModelLoader.make_model_registry_str()
         print(model_str)
         sys.exit()

    # overwrite config_fn is argument passed
    if arg_overwrite and args.config_fn is not None:
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
    m = overwrite_defaults(m, args, 'model_type')
    m = overwrite_defaults(m, args, 'model_variant')
    m = overwrite_defaults(m, args, 'num_char')
    m = overwrite_defaults(m, args, 'sim_logging')
    m = overwrite_defaults(m, args, 'start_idx')
    m = overwrite_defaults(m, args, 'end_idx')
    m = overwrite_defaults(m, args, 'stop_time')
    m = overwrite_defaults(m, args, 'min_num_taxa')
    m = overwrite_defaults(m, args, 'max_num_taxa')
    m = overwrite_defaults(m, args, 'tree_width')
    m = overwrite_defaults(m, args, 'save_phyenc_csv')
    m = overwrite_defaults(m, args, 'tree_width_cats')
    m = overwrite_defaults(m, args, 'num_epochs')
    m = overwrite_defaults(m, args, 'batch_size')
    m = overwrite_defaults(m, args, 'prop_test')
    m = overwrite_defaults(m, args, 'prop_validation')
    m = overwrite_defaults(m, args, 'prop_calibration')
    m = overwrite_defaults(m, args, 'cpi_coverage')
    m = overwrite_defaults(m, args, 'loss')
    m = overwrite_defaults(m, args, 'optimizer')
    m = overwrite_defaults(m, args, 'pred_prefix')
    m = overwrite_defaults(m, args, 'plot_train_color')
    m = overwrite_defaults(m, args, 'plot_test_color')
    m = overwrite_defaults(m, args, 'plot_val_color')
    m = overwrite_defaults(m, args, 'plot_aux_color')
    m = overwrite_defaults(m, args, 'plot_label_color')
    m = overwrite_defaults(m, args, 'plot_pred_color')         

    # return new args
    return m.args


#-----------------------------------------------------------------------------------------------------------------#

###################
# GENERAL HELPERS #
###################

def make_symm(m):
    """
    Makes a matrix symmetric by copying the upper triangle to the lower triangle.

    Args:
        m (numpy.ndarray): The input matrix.

    Returns:
        numpy.ndarray: The symmetric matrix.
    """
    d = np.diag(m)  # Extracts the diagonal elements of the matrix
    m = np.triu(m)  # Extracts the upper triangle of the matrix
    m = m + m.T  # Adds the transposed upper triangle to the original upper triangle
    np.fill_diagonal(m, d)  # Restores the original diagonal elements
    return m

def sort_binary_vectors(binary_vectors):
    """
    Sorts a list of binary vectors.

    The binary vectors are sorted first based on the number of "on" bits, and then from left to right in terms of which bits are "on".

    Args:
        binary_vectors (List[List[int]]): The list of binary vectors to be sorted.

    Returns:
        List[List[int]]: The sorted list of binary vectors.
    """
    def count_ones(binary_vector):
        """
        Counts the number of "on" bits in a binary vector.

        Args:
            binary_vector (List[int]): The binary vector.

        Returns:
            int: The count of "on" bits.
        """
        return sum(binary_vector)

    sorted_vectors = sorted(binary_vectors, key=count_ones)

    for i in range(len(sorted_vectors)):
        for j in range(i+1, len(sorted_vectors)):
            if count_ones(sorted_vectors[j]) == count_ones(sorted_vectors[i]):
                for k in range(len(sorted_vectors[i])):
                    if sorted_vectors[i][k] != sorted_vectors[j][k]:
                        if sorted_vectors[j][k] > sorted_vectors[i][k]:
                            sorted_vectors[i], sorted_vectors[j] = sorted_vectors[j], sorted_vectors[i]
                        break

    return sorted_vectors

def powerset(iterable):
    """
    Generates all possible subsets (powerset) of the given iterable.

    Args:
        iterable: An iterable object.

    Returns:
        generator: A generator that yields each subset.
    """
    s = list(iterable)  # Convert the iterable to a list
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def find_tree_width(num_taxa:int, max_taxa:list[int]):
    """Finds the CPSV width.

    Returns the smallest suitable compact phylogenetic-state vector
    representation such that num_taxa <= val for val in max_taxa.
    Returns 0 if num_taxa == 0.
    Returns -1 if num_taxa > max_taxa[-1].

    Args:
        num_taxa (int): the number of taxa in the raw dataset
        max_taxa (list[int]):  a list of tree widths for CPSV encoding
    
    Returns:
        int: The smallest suitable tree width encoding
    """
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



#-----------------------------------------------------------------------------------------------------------------#


################
# FILE HELPERS #
################

def write_to_file(s: str, fn: str) -> None:
    """Writes a string to a file.

    Args:
        s (str): The string to write.
        fn (str): The file name or path to write the string to.

    Returns:
        None
    """
    f = open(fn, 'w')
    f.write(s)
    f.close()
    return


def read_tree(tre_fn):
    """Reads a phylogenetic tree from a file.

    Args:
        tre_fn (str): The file name or path of the tree file.

    Returns:
        dp.Tree or None: The parsed phylogenetic tree object, or None if the tree cannot be read.

    Raises:
        FileNotFoundError: If the tree file at `tre_fn` does not exist.
    """
    if not os.path.exists(tre_fn):
        raise FileNotFoundError(f'Could not find tree file at {tre_fn}')

    phy = None
    for schema in ['newick', 'nexus']:
        try:
            phy_tmp = dp.Tree.get(path=tre_fn, schema=schema)
        except:
            phy_tmp = None
        else:
            if phy_tmp is not None:
                phy = phy_tmp
    return phy




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
#def make_prune_phy(tre_fn, prune_fn):
def make_prune_phy(phy, prune_fn):
    # read tree
    # phy_ = dp.Tree.get(path=tre_fn, schema='newick')
    # copy input tree
    phy_ = copy.deepcopy(phy)
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
        return None
    else:
        # prune non-extant taxa
        phy_.prune_taxa_with_labels( drop_taxon_labels )
        # write pruned tree
        phy_.write(path=prune_fn, schema='newick')
        return phy_



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

def make_clean_phyloenc_str(x):
    s = np.array2string(x, separator=',', max_line_width=1e200, threshold=1e200, edgeitems=1e200, precision=10, floatmode='maxprec')
    s = re.sub(r'[\[\]]', '', string=s)
    s = re.sub(r',\n ', '\n', string=s)
    s = s + '\n'
    return s

def clean_scientific_notation(s):
    return re.sub( '\.0+E\+0+', '', s)



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

