#!/usr/bin/env python
"""
utilities
===========
Miscellaneous helper functions phyddle uses for pipeline steps.

Authors:   Michael Landis and Ammon Thompson
Copyright: (c) 2022-2023, Michael Landis and Ammon Thompson
License:   MIT
"""

# standard packages
import argparse
import copy
import importlib
import os
import pkg_resources
import platform
import re
import sys
from datetime import datetime
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
    def __init__(self, idx, r=0.0, n=None, g=None, ix=None, jx=None):
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
    def make_str(self):
        """
        Creates a string representation of the event.

        Returns:
            str: The string representation of the event.
        """
        s = 'Event({name},{group},{rate},{idx})'.format(name=self.name, group=self.group, rate=self.rate, idx=self.idx)        
        #s += ')'
        return s
    
    # representation string
    def __repr__(self):
        """
        Returns the representation of the event.

        Returns:
            str: The representation of the event.
        """
        return self.make_str()
    
    # print string
    def __str__(self):
        """
        Returns the string representation of the event.

        Returns:
            str: The string representation of the event.
        """
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
       
        # done
        return

    def make_str(self):
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
    def __repr__(self):
        """
        Returns the representation of the state space.

        Returns:
            str: The representation of the state space.
        """
        return self.make_str()
    # print string
    def __str__(self):
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



#------------------------------------------------------------------------------#

###################
# CONFIG LOADER   #
###################

def load_config(config_fn,
                arg_overwrite=True,
                args=None):
    """
    Loads the configuration.

    Args:
        config_fn (str): The config file name.
        arg_overwrite (bool, optional): Whether to overwrite config file settings with arguments. Defaults to True.

    Returns:
        dict: The loaded configuration.
    """

    # use command line sys.argv if no args provided
    if args is None:
        args = sys.argv[1:]

    # argument parsing
    parser = argparse.ArgumentParser(description='phyddle pipeline config') #,
                                     #formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # basic settings
    parser.add_argument('-c', '--cfg',          dest='config_fn', type=str, help='Config file name', metavar='')
    #parser.add_argument('-f', '--force',        action='store_true', help='Arguments override config file settings')
    parser.add_argument('-p', '--proj',         dest='proj', type=str, help='Project name used as directory across pipeline stages', metavar='')
    parser.add_argument('-s', '--step',         dest='step', type=str, help='Pipeline step(s) defined with (S)imulate, (F)ormat, (T)rain, (E)stimate, (P)lot, or (A)ll', metavar='')
    parser.add_argument('-v', '--verbose',      dest='verbose', type=bool, help='Verbose output to screen? (recommended)', metavar='')
    parser.add_argument('--make_cfg',           action='store_true', help='Write default config file to \'config_default.py\'?')
    # processor settings
    parser.add_argument('--use_parallel',       dest='use_parallel', type=bool, help='Use parallelization? (recommended)', metavar='')
    parser.add_argument('--num_proc',           dest='num_proc', type=int, help='How many cores for multiprocessing? (e.g. 4 uses 4, -2 uses all but 2)', metavar='')
    # directory settings
    parser.add_argument('--sim_dir',            dest='sim_dir', type=str, help='Directory for raw simulated data', metavar='')
    parser.add_argument('--fmt_dir',            dest='fmt_dir', type=str, help='Directory for tensor-formatted simulated data', metavar='')
    parser.add_argument('--trn_dir',            dest='trn_dir', type=str, help='Directory for trained networks and training predictions', metavar='')
    parser.add_argument('--est_dir',            dest='est_dir', type=str, help='Directory for new datasets and predictions', metavar='')
    parser.add_argument('--plt_dir',            dest='plt_dir', type=str, help='Directory for plotted results', metavar='')
    parser.add_argument('--log_dir',            dest='log_dir', type=str, help='Directory for logs of analysis metadata', metavar='')
    # model settings
    parser.add_argument('--show_models',        action='store_true', help='Print all available model types and variants?')
    parser.add_argument('--model_type',         dest='model_type', type=str, help='Model type', metavar='')
    parser.add_argument('--model_variant',      dest='model_variant', type=str, help='Model variant', metavar='')
    parser.add_argument('--num_char',           dest='num_char', type=int, help='Number of characters', metavar='')
    # simulation settings
    parser.add_argument('--sim_method',         dest='sim_method', type=str, choices=['command', 'master'], help='Simulation method', metavar='')
    parser.add_argument('--sim_command',        dest='sim_command', type=str, help='Simulation command (when sim_method==\'command\')', metavar='')
    parser.add_argument('--sim_logging',        dest='sim_logging', type=str, choices=['clean', 'verbose', 'compress'], help='Simulation logging style', metavar='')
    parser.add_argument('--start_idx',          dest='start_idx', type=int, help='Start index for simulation', metavar='')
    parser.add_argument('--end_idx',            dest='end_idx', type=int, help='End index for simulation', metavar='')
    parser.add_argument('--stop_time',          dest='stop_time', type=float, help='Maximum duration of evolution for each simulation', metavar='')
    parser.add_argument('--min_num_taxa',       dest='min_num_taxa', type=int, help='Minimum number of taxa for each simulation', metavar='')
    parser.add_argument('--max_num_taxa',       dest='max_num_taxa', type=int, help='Maximum number of taxa for each simulation', metavar='')
    # formatting settings
    parser.add_argument('--tree_type',          dest='tree_type', type=str, choices=['extant', 'serial'], help='Type of tree', metavar='')
    parser.add_argument('--tree_width_cats',    dest='tree_width_cats', type=int, help='The phylo-state tensor widths for formatting training datasets, space-delimited', metavar='')
    parser.add_argument('--tree_encode_type',   dest='tree_encode_type', type=str, choices=['height_only', 'height_brlen'], help='Method for encoding branch length info in tensor', metavar='')
    parser.add_argument('--char_encode_type',   dest='char_encode_type', type=str, choices=['one_hot', 'integer', 'numeric'], help='Method for encoding character states in tensor', metavar='')
    parser.add_argument('--chardata_format',    dest='chardata_format', type=str, choices=['nexus', 'csv'], help='Input format for character matrix data', metavar='')
    parser.add_argument('--tensor_format',      dest='tensor_format', type=str, choices=['hdf5', 'csv'], help='Output format for storing tensors of training dataset', metavar='')
    parser.add_argument('--save_phyenc_csv',    dest='save_phyenc_csv', type=bool, help='Save encoded phylogenetic tensor encoding to csv?', metavar='')
    # training settings
    parser.add_argument('--trn_objective',      dest='trn_objective', type=str, choices=['param_est', 'model_test'], help='Objective of training procedure', metavar='')
    parser.add_argument('--tree_width',         dest='tree_width', type=int, help='The phylo-state tensor width dataset used for a neural network', metavar='')
    parser.add_argument('--num_epochs',         dest='num_epochs', type=int, help='Number of training epochs', metavar='')
    parser.add_argument('--batch_size',         dest='batch_size', type=int, help='Training batch sizes', metavar='')
    parser.add_argument('--prop_test',          dest='prop_test', type=float, help='Proportion of data used as test examples (demonstrate trained network performance)', metavar='')
    parser.add_argument('--prop_validation',    dest='prop_validation', type=float, help='Proportion of data used as validation examples (diagnose network overtraining)', metavar='')
    parser.add_argument('--prop_calibration',   dest='prop_calibration', type=float, help='Proportion of data used as calibration examples (calibrate conformal prediction intervals)', metavar='')
    parser.add_argument('--cpi_coverage',       dest='cpi_coverage', type=float, help='Expected coverage percent for calibrated prediction intervals (CPIs)', metavar='')
    parser.add_argument('--cpi_asymmetric',     dest='cpi_asymmetric', type=bool, help='Use asymmetric (True) or symmetric (False) adjustments for CPIs?', metavar='')
    parser.add_argument('--loss',               dest='loss', type=str, help='Loss function used as optimization criterion', metavar='')
    parser.add_argument('--optimizer',          dest='optimizer', type=str, help='Method used for optimizing neural network', metavar='')
    # prediction settings
    parser.add_argument('--est_prefix',         dest='est_prefix', type=str, help='Predict results for this dataset', metavar='')
    # plotting settings
    parser.add_argument('--plot_train_color',   dest='plot_train_color', type=str, help='Plotting color for training data elements', metavar='')
    parser.add_argument('--plot_label_color',   dest='plot_label_color', type=str, help='Plotting color for training label elements', metavar='')
    parser.add_argument('--plot_test_color',    dest='plot_test_color', type=str, help='Plotting color for test data elements', metavar='')
    parser.add_argument('--plot_val_color',     dest='plot_val_color', type=str, help='Plotting color for validation data elements', metavar='')
    parser.add_argument('--plot_aux_color',     dest='plot_aux_color', type=str, help='Plotting color for auxiliary input data elements', metavar='')
    parser.add_argument('--plot_est_color',     dest='plot_est_color', type=str, help='Plotting color for new estimation elements', metavar='')


    # parse arguments
    args = parser.parse_args(args)
    
    # print models & exit
    if args.show_models:
         import model_loader
         model_str = model_loader.make_model_registry_str()
         print(model_str)
         sys.exit()

    # make default config
    if args.make_cfg:
        make_default_config()
        print('Created default config as \'config_default.py\' ...')
        sys.exit()

    # overwrite config_fn is argument passed
    if arg_overwrite and args.config_fn is not None:
        config_fn = args.config_fn
    config_fn = config_fn.rstrip('.py')

    # get config from file
    m = importlib.import_module(config_fn)
    
    # update arguments from defaults, when provided
    def overwrite_defaults(m, args, var):
        x = getattr(args, var)
        if x is not None:
            # if args.force == True:
            #     m.args[var] = x
            # elif var not in m.args:
            #     m.args[var] = x
            m.args[var] = x
        return m
    
    m = overwrite_defaults(m, args, 'proj')
    m = overwrite_defaults(m, args, 'step')
    m = overwrite_defaults(m, args, 'use_parallel')
    m = overwrite_defaults(m, args, 'verbose')
    m = overwrite_defaults(m, args, 'num_proc')
    m = overwrite_defaults(m, args, 'sim_dir')
    m = overwrite_defaults(m, args, 'fmt_dir')
    m = overwrite_defaults(m, args, 'trn_dir')
    m = overwrite_defaults(m, args, 'est_dir')
    m = overwrite_defaults(m, args, 'plt_dir')
    m = overwrite_defaults(m, args, 'log_dir')
    m = overwrite_defaults(m, args, 'model_type')
    m = overwrite_defaults(m, args, 'model_variant')
    m = overwrite_defaults(m, args, 'num_char')
    m = overwrite_defaults(m, args, 'sim_method')
    m = overwrite_defaults(m, args, 'sim_command')
    m = overwrite_defaults(m, args, 'sim_logging')
    m = overwrite_defaults(m, args, 'start_idx')
    m = overwrite_defaults(m, args, 'end_idx')
    m = overwrite_defaults(m, args, 'stop_time')
    m = overwrite_defaults(m, args, 'min_num_taxa')
    m = overwrite_defaults(m, args, 'max_num_taxa')
    m = overwrite_defaults(m, args, 'tree_width')
    m = overwrite_defaults(m, args, 'save_phyenc_csv')
    m = overwrite_defaults(m, args, 'tree_width_cats')
    m = overwrite_defaults(m, args, 'tree_encode_type')
    m = overwrite_defaults(m, args, 'char_encode_type')
    m = overwrite_defaults(m, args, 'tensor_format')
    m = overwrite_defaults(m, args, 'chardata_format')
    m = overwrite_defaults(m, args, 'trn_objective')
    m = overwrite_defaults(m, args, 'num_epochs')
    m = overwrite_defaults(m, args, 'batch_size')
    m = overwrite_defaults(m, args, 'prop_test')
    m = overwrite_defaults(m, args, 'prop_validation')
    m = overwrite_defaults(m, args, 'prop_calibration')
    m = overwrite_defaults(m, args, 'cpi_coverage')
    m = overwrite_defaults(m, args, 'cpi_asymmetric')
    m = overwrite_defaults(m, args, 'loss')
    m = overwrite_defaults(m, args, 'optimizer')
    m = overwrite_defaults(m, args, 'est_prefix')
    m = overwrite_defaults(m, args, 'plot_train_color')
    m = overwrite_defaults(m, args, 'plot_test_color')
    m = overwrite_defaults(m, args, 'plot_val_color')
    m = overwrite_defaults(m, args, 'plot_aux_color')
    m = overwrite_defaults(m, args, 'plot_label_color')
    m = overwrite_defaults(m, args, 'plot_est_color')         

    # check arguments are valid
    check_args(m.args)

    # set steps & projects
    m.args = add_step_proj(m.args) #['step'], m.args['proj'])
    
    if m.args['step'] == 'A':
        m.args['step'] = ['S', 'F', 'T', 'E', 'P']
    else:
        m.args['step'] = [ i for i in m.args['step'] ]

    # add unique ID
    m.args['job_id'] = generate_random_hex_string(16)
    
    # print header?
    verbose = m.args['verbose']
    if verbose:
        print(phyddle_header('title'))

    # return new args
    return m.args

def check_args(args):
    """
    Checks if the given arguments meet certain conditions.

    Parameters:
    args (dict): A dictionary containing the arguments.

    Raises:
    AssertionError: If any of the conditions are not met.
    """
    # string values
    assert all([s in 'ASFTEP' for s in args['step']])
    assert args['sim_method']        in ['command', 'master']
    assert args['sim_logging']       in ['clean', 'verbose', 'compress']
    assert args['tree_type']         in ['serial', 'extant']
    assert args['tree_encode_type']  in ['height_only', 'height_brlen']
    assert args['char_encode_type']  in ['one_hot', 'integer', 'numeric']
    assert args['tensor_format']     in ['csv', 'hdf5']
    assert args['chardata_format']   in ['csv', 'nexus']
    assert args['trn_objective']     in ['param_est', 'model_test']
    
    # numerical values
    assert args['start_idx'] >= 0
    assert args['end_idx'] >= 0
    assert args['start_idx'] <= args['end_idx']
    assert args['min_num_taxa'] >= 0
    assert args['max_num_taxa'] >= 0
    assert args['min_num_taxa'] <= args['max_num_taxa']
    assert args['num_states'] > 0
    assert args['num_char'] > 0
    assert args['num_epochs'] > 0
    assert args['batch_size'] > 0
    assert args['cpi_coverage'] >= 0. and args['cpi_coverage'] <= 1.
    assert args['prop_test'] >= 0. and args['prop_test'] <= 1.
    assert args['prop_validation'] >= 0. and args['prop_validation'] <= 1.
    assert args['prop_calibration'] >= 0. and args['prop_calibration'] <= 1.
    assert len(args['tree_width_cats']) > 0
    for i in range(len(args['tree_width_cats'])):
        assert args['tree_width_cats'][i] > 0
    assert args['tree_width'] in args['tree_width_cats']

    return

def add_step_proj(args): #steps, proj):
    
    # get relevant args
    steps = args['step']
    proj = args['proj']
    
    d_map = { 'S': ('sim', 'simulate'),
              'F': ('fmt', 'format'),
              'T': ('trn', 'train'),
              'E': ('est', 'estimate'),
              'P': ('plt', 'plot'),
              'L': ('log', 'log') }
    
    # # treat proj as the global project name
    # # if it contains no split tokens
    # if ':' not in proj and ',' not in proj:
    #     proj = f'A:{proj}'
    
    # parse input string
    d_toks = {}
    proj_toks = proj.split(',')
    for p in proj_toks:
        if ':' not in p:
            d_toks['A'] = p
        else:
            k,v = p.split(':')
            d_toks[k] = v

    # handle all-step (*) first
    d_arg = {}
    if 'A' in d_toks.keys():
        steps = 'SFTEPL'
        for i in ['S', 'F', 'T', 'E', 'P', 'L']:
            k = d_map[i][0]
            d_arg[k] = d_toks['A']
        
    # overwrite with named steps
    k_change = [ k for k in d_toks.keys() if k in 'SFTEPL' ]
    for k in k_change:
        d_arg[ d_map[k][0] ] = d_toks[k]
    
    # verify all steps are covered
    for s in steps:
        if d_map[s][0] not in d_arg.keys():
            raise ValueError(f"Step {s} ({d_map[s][1]}) has no assigned project name")

    for k in d_arg.keys():
        k_str = k + '_proj'
        args[k_str] = d_arg[k]

    return args

def make_default_config():
    # can we have run_phyddle use this if no cfg file is provided??
    s = """
#==============================================================================#
# Default phyddle config file                                                  #
#==============================================================================#

# external import
import scipy.stats
import scipy as sp

# helper variables
num_char = 3
num_states = 2

args = {

    #-------------------------------#
    # Project organization          #
    #-------------------------------#
    'proj'    : 'my_project',               # project name(s)
    'step'    : 'SFTEP',                    # step(s) to run
    'verbose' : True,                       # print verbose phyddle output?
    'sim_dir' : '../workspace/simulate',    # directory for simulated data
    'fmt_dir' : '../workspace/format',      # directory for tensor-formatted data
    'trn_dir' : '../workspace/train',       # directory for trained network
    'plt_dir' : '../workspace/plot',        # directory for plotted figures
    'est_dir' : '../workspace/estimate',    # directory for predictions on new data
    'log_dir' : '../workspace/log',         # directory for analysis logs

    #-------------------------------#
    # Multiprocessing               #
    #-------------------------------#
    'use_parallel'   : True,                # use multiprocessing to speed up jobs?
    'num_proc'       : -2,                  # how many CPUs to use (-2 means all but 2)

    #-------------------------------#
    # Model Configuration           #
    #-------------------------------#
    'model_type'         : 'geosse',        # model type defines general states and events
    'model_variant'      : 'equal_rates',   # model variant defines rate assignments
    'num_char'           : num_char,        # number of evolutionary characters
    'num_states'         : num_states,      # number of states per character
    'rv_fn'              : {                # distributions for model parameters
        'w': sp.stats.expon.rvs,
        'e': sp.stats.expon.rvs,
        'd': sp.stats.expon.rvs,
        'b': sp.stats.expon.rvs
    },
    'rv_arg'             : {                # loc/scale/shape for model parameter dists
        'w': { 'scale' : 0.2 },
        'e': { 'scale' : 0.1 },
        'd': { 'scale' : 0.1 },
        'b': { 'scale' : 0.5 }
    },

    #-------------------------------#
    # Simulate Step settings        #
    #-------------------------------#
    'sim_method'        : 'master',         # command, master, [phylojunction], ...
    'sim_command'       : 'beast',          # exact command string, argument is output file prefix
    'sim_logging'       : 'verbose',        # verbose, compressed, or clean
    'start_idx'         : 0,                # first simulation replicate index
    'end_idx'           : 1000,             # last simulation replicate index
    'sample_population' : ['S'],            # name of population to sample
    'stop_time'         : 10,               # time to stop simulation
    'min_num_taxa'      : 10,               # min number of taxa for valid sim
    'max_num_taxa'      : 500,              # max number of taxa for valid sim

    #-------------------------------#
    # Format Step settings          #
    #-------------------------------#
    'tree_type'         : 'extant',         # use model with serial or extant tree
    'chardata_format'   : 'nexus',
    'tree_width_cats'   : [ 200, 500 ],     # tree width categories for phylo-state tensors
    'tree_encode_type'  : 'height_brlen',   # how to encode phylo brlen? height_only or height_brlen
    'char_encode_type'  : 'integer',        # how to encode discrete states? one_hot or integer
    'param_pred'        : [                 # model parameters to predict (labels)
        'w_0', 'e_0', 'd_0_1', 'b_0_1'
    ],
    'param_data'        : [],               # model parameters that are known (aux. data)
    'tensor_format'     : 'hdf5',           # save as compressed HDF5 or raw csv
    'save_phyenc_csv'   : False,            # save intermediate phylo-state vectors to file

    #-------------------------------#
    # Train Step settings           #
    #-------------------------------#
    'trn_objective'     : 'param_est',      # what is the learning task? param_est or model_test
    'tree_width'        : 500,              # tree width category used to train network
    'num_epochs'        : 20,               # number of training intervals (epochs)
    'prop_test'         : 0.05,             # proportion of sims in test dataset
    'prop_validation'   : 0.05,             # proportion of sims in validation dataset
    'prop_calibration'  : 0.20,             # proportion of sims in CPI calibration dataset
    'cpi_coverage'      : 0.95,             # coverage level for CPIs
    'cpi_asymmetric'    : True,             # upper/lower (True) or symmetric (False) CPI adjustments
    'batch_size'        : 128,              # number of samples in each training batch
    'loss'              : 'mse',            # loss function for learning
    'optimizer'         : 'adam',           # optimizer for network weight/bias parameters
    'metrics'           : ['mae', 'acc'],   # recorded training metrics

    #-------------------------------#
    # Estimate Step settings        #
    #-------------------------------#
    'est_prefix'     : 'new.1',             # prefix for new dataset to predict

    #-------------------------------#
    # Plot Step settings            #
    #-------------------------------#
    'plot_train_color'      : 'blue',       # plot color for training data
    'plot_test_color'       : 'purple',     # plot color for test data
    'plot_val_color'        : 'red',        # plot color for validation data
    'plot_aux_color'        : 'green',      # plot color for input auxiliary data
    'plot_label_color'      : 'orange',     # plot color for labels (params)
    'plot_est_color'        : 'black'       # plot color for estimated data/values

}
"""
    f = open('config_default.py', 'w')
    f.write(s)
    f.close()
    return

def generate_random_hex_string(length):
    """
    Generates a random hex string of a given length.

    Parameters:
    length (int): The length of the hex string.

    Returns:
    str: The generated random hex string.
    """
    hex_chars = '0123456789abcdef'
    random_indices = np.random.randint(0, len(hex_chars), size=length)
    hex_string = ''.join(hex_chars[i] for i in random_indices)
    return hex_string

#------------------------------------------------------------------------------#

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


def find_tree_width(num_taxa, max_taxa):
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
    #return -2


def get_num_tree_row(tree_type, tree_encode_type):
    if tree_type == 'serial':
        num_tree_row = 2
    elif tree_type == 'extant':
        num_tree_row = 1

    if tree_encode_type == 'height_only':
        num_tree_row += 0
    elif tree_encode_type == 'height_brlen':
        num_tree_row += 2

    return num_tree_row

def get_num_char_row(state_encode_type, num_char, num_states):
        
    if state_encode_type == 'integer':
        num_char_row = num_char
    elif state_encode_type == 'one_hot':
        num_char_row = num_char * num_states

    return num_char_row


#------------------------------------------------------------------------------#


################
# FILE HELPERS #
################

def write_to_file(s, fn):
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




#------------------------------------------------------------------------------#

#####################
# FORMAT CONVERTERS #
#####################


def convert_csv_to_array(dat_fn, char_encode_type, num_states=None):
    if char_encode_type == 'numeric':
        dat = convert_csv_to_numeric_array(dat_fn, num_states)
    elif char_encode_type == 'one_hot':
        dat = convert_csv_to_onehot_array(dat_fn, num_states)
    else:
        return NotImplementedError
    return dat



def convert_csv_to_numeric_array(dat_fn, pca_compress=None):
    """Converts a csv file to an integer-encoded pandas DataFrame.

    """
    # read as pandas
    dat = pd.read_csv(dat_fn, delimiter=',', index_col=0, header=None).T 

    # return
    return dat

def convert_csv_to_onehot_array(dat_fn, num_states):
    """Converts a csv file to an integer-encoded pandas DataFrame.

    """
    # read data
    dat_raw = pd.read_csv(dat_fn, delimiter=',', index_col=0, header=None).T

    # get num taxa (columns)
    num_taxa = dat_raw.shape[1]

    # check/unify number of state per row
    if type(num_states) == 'int':
        num_states = [ num_states ] * dat_raw.shape[0]
    assert(dat_raw.shape[0] == len(num_states))
    assert(all([type(i)=='int' for i in num_states]))

    # make csv
    num_rows = sum(num_states)
    zero_data = np.zeros(shape=(num_rows,num_taxa))
    dat = pd.DataFrame(zero_data, columns=dat_raw.columns)

    # do one-hot encoding
    num_states_offset = 0
    for j in range(num_rows):
        for i in range(num_taxa):
            k = dat_raw.iloc[j,i] + num_states_offset
            dat.iloc[k,i] = 1
        num_states_offset += num_states[j]

    # done!
    return dat


def convert_nexus_to_array(dat_fn, char_encode_type, num_states=None):
    """
    Convert Nexus file to array format.

    Parameters:
    dat_fn (str): The file name of the Nexus file.
    char_encode_type (str): The type of character encoding to use. Valid options are "integer" or "one_hot".
    num_states (int, optional): The number of states. Only applicable if char_encode_type is "one_hot".

    Returns:
    dat (array): The converted array.
    """
    if char_encode_type == 'integer':
        dat = convert_nexus_to_integer_array(dat_fn)
    elif char_encode_type == 'one_hot':
        dat = convert_nexus_to_onehot_array(dat_fn, num_states)
    else:
        return NotImplementedError

    return dat

def convert_nexus_to_integer_array(dat_fn):
    """Converts a NEXUS file to an integer-encoded pandas DataFrame.

    Reads the NEXUS file specified by `dat_fn`, extracts the data matrix, and constructs a pandas DataFrame where rows represent character states and columns represent taxa.

    Args:
        dat_fn (str): The file name or path of the NEXUS file.

    Returns:
        pd.DataFrame: The pandas DataFrame representing the data matrix.

    Raises:
        FileNotFoundError: If the NEXUS file at `dat_fn` does not exist.
    """
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
    df = pd.DataFrame(dat, columns=taxon_names)
    
    return df

def convert_nexus_to_onehot_array(dat_fn, num_states):
    """Converts a NEXUS file to a one-hot encoded pandas DataFrame.

    Reads the NEXUS file specified by `dat_fn`, extracts the data matrix, and constructs a pandas DataFrame where rows represent character states and columns represent taxa.

    Args:
        dat_fn (str): The file name or path of the NEXUS file.
        num_states (int, optional): Number of states to one-hot encode. Learned from length of symbols when None. Defaults to None.

    Returns:
        pd.DataFrame: The pandas DataFrame representing the data matrix.

    Raises:
        FileNotFoundError: If the NEXUS file at `dat_fn` does not exist.


    Character state-vector with one-hot encoding
    Example:
    Let num_char = 2 and num_state = 3 for states {0, 1, 2}
                ---------------------
        Char:   | 0  0  0 | 1  1  1 |
        State:  | 0  1  2 | 0  1  2 |
                |---------|---------|
        Encode: | 0  1  2 | 3  4  5 |
                ---------------------
    Taxon with state-vector "20" -> "001100"
    Taxon with state-vector "12" -> "010001"
    etc.
    """
    # read file
    f = open(dat_fn, 'r')
    lines = f.readlines()
    f.close()

    # helper variables
    found_matrix = False
    num_taxa    = 0
    num_char    = 0
    num_one_hot = 0
    taxon_idx   = 0
    taxon_names = []
    
    #print('\n'.join(lines))
    # process file
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
                    num_one_hot = num_states * num_char
            dat = np.zeros((num_one_hot, num_taxa), dtype='int')

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
                # Taxon name
                name = tok[0]
                taxon_names.append(name)
                # One-hot encoding
                state = tok[1]
                v = [ int(z) for z in state ]
                #print(v)
                for i,j in enumerate(v):
                    #print('  ',i,j)
                    state_idx = i * num_states + j
                    #print(state_idx)
                    dat[state_idx,taxon_idx] = 1
                taxon_idx += 1

    # construct data frame
    df = pd.DataFrame(dat, columns=taxon_names)
    
    return df

def convert_phy2dat_nex(phy_nex_fn, int2vec):
    """
    Converts a phylogenetic tree in NHX format to a NEXUS file with taxon-state data.

    Reads the phylogenetic tree file in NHX format specified by `phy_nex_fn` and converts it to a NEXUS file containing taxon-state data. The binary state representations are based on the provided `int2vec` mapping.

    Args:
        phy_nex_fn (str): The file name or path of the phylogenetic tree file in NHX format.
        int2vec (List[int]): The mapping of integer states to binary state vectors.

    Returns:
        str: The NEXUS file content as a string.

    Raises:
        FileNotFoundError: If the phylogenetic tree file at `phy_nex_fn` does not exist.
    """

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

def make_prune_phy(phy, prune_fn):
    """Prunes a phylogenetic tree by removing non-extant taxa and writes the pruned tree to a file.

    The function takes a phylogenetic tree `phy` and a file name `prune_fn` as input. It prunes the tree by removing non-extant taxa and writes the pruned tree to the specified file.

    Args:
        phy (Tree): The input phylogenetic tree.
        prune_fn (str): The file name or path to write the pruned tree.

    Returns:
        Tree or None: The pruned phylogenetic tree if pruning is successful, or None if the pruned tree would have fewer than two leaf nodes (invalid tree).

    Raises:
        None.
    """
    # copy input tree
    phy_ = copy.deepcopy(phy)
    # compute all root-to-node distances
    root_distances = phy_.calc_node_root_distances()
    # find tree height (max root-to-node distance)
    tree_height = np.max( root_distances )
    # tips are considered "at present" if age is within 0.0001 * tree_height
    tol = tree_height * 1e-5
    # create empty dictionary
    d = {}
    # loop through all leaf nodes
    leaf_nodes = phy_.leaf_nodes()
    for i,nd in enumerate(leaf_nodes):
        # convert root-distances to ages
        age = tree_height - nd.root_distance
        nd.annotations.add_new('age', age)
        # ultrametricize ages for extant taxa
        if age < tol:
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

def settings_to_str(settings, taxon_category):
    """
    Convert settings dictionary and taxon category to a string representation.

    This function takes a settings dictionary and a taxon category and converts them into a comma-separated string representation. The resulting string includes the keys and values of the settings dictionary, as well as the taxon category.

    Args:
        settings (dict): The settings dictionary.
        taxon_category (str): The taxon category.

    Returns:
        str: The string representation of the settings and taxon category.
    """
    s = 'setting,value\n'
    s += 'model_name,' + settings['model_name'] + '\n'
    s += 'model_type,' + settings['model_type'] + '\n'
    s += 'replicate_index,' + str(settings['replicate_index']) + '\n'
    s += 'taxon_category,' + str(taxon_category) + '\n'
    return s

def param_dict_to_str(params):
    """
    Convert parameter dictionary to two string representations.

    This function takes a parameter dictionary and converts it into two string representations. The resulting strings includes the parameter names, indices, and values. The first representation is column-based, the second representation is row-based.

    Args:
        params (dict): The parameter dictionary.

    Returns:
        tuple: A tuple of two strings. The first string represents the parameter values with indices, and the second string represents the parameter names.
    """
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
    """
    Convert a list of Event objects to a pandas DataFrame.

    This function takes a list of Event objects and converts it into a pandas DataFrame. Each Event object represents a row in the resulting DataFrame, with the Event attributes mapped to columns.

    Args:
        events (list): A list of Event objects.

    Returns:
        pandas.DataFrame: The resulting DataFrame with columns 'name', 'group', 'i', 'j', 'k', 'reaction', and 'rate'.
    """
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
    """
    Convert a States object to a pandas DataFrame.

    This function takes a States object and converts it into a pandas DataFrame. The States object contains information about the state space, and the resulting DataFrame has columns 'lbl', 'int', 'set', and 'vec', representing the labels, integer representations, set representations, and vector representations of the states, respectively.

    Args:
        states (States): The States object to convert to a DataFrame.

    Returns:
        pandas.DataFrame: The resulting DataFrame with columns 'lbl', 'int', 'set', and 'vec'.
    """
    df = pd.DataFrame({
        'lbl' : states.int2lbl,
        'int' : states.int2int,
        'set' : states.int2set,
        'vec' : states.int2vec
    })
    return df

# make matrix with parameter values, lower-bounds, upper-bounds: 3D->2D
def make_param_VLU_mtx(A, param_names):
    """
    Convert a parameter matrix to a pandas DataFrame with combined header indices.

    This function takes a parameter matrix A and a list of parameter names and creates a pandas DataFrame with combined header indices. The resulting DataFrame has columns representing different statistics (value, lower, upper), replicated indices, and parameters. The parameter names and statistics are combined to form the column headers.

    Args:
        A (numpy.ndarray): The parameter matrix.
        param_names (list): A list of parameter names.

    Returns:
        pandas.DataFrame: The resulting DataFrame with combined header indices.
    """
    
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
    """
    Convert a numpy array to a clean string representation.

    This function takes a numpy array `x` and converts it to a clean string representation. The resulting string is obtained by formatting the array with a comma separator, removing the square brackets, and replacing line breaks and unnecessary whitespace characters. The string representation is useful for displaying or saving the array in a clean and readable format.

    Args:
        x (numpy.ndarray): The numpy array to convert.

    Returns:
        str: The clean string representation of the numpy array.
    """
    s = np.array2string(x, separator=',', max_line_width=1e200, threshold=1e200, edgeitems=1e200, precision=10, floatmode='maxprec')
    s = re.sub(r'[\[\]]', '', string=s)
    s = re.sub(r',\n ', '\n', string=s)
    s = s + '\n'
    return s

def clean_scientific_notation(s):
    """
    Clean up a string representation of a number in scientific notation.

    This function takes a string `s` representing a number in scientific notation and removes unnecessary characters that indicate zero values. The resulting string represents the number without trailing zeros in the exponent.

    Args:
        s (str): The string representation of a number in scientific notation.

    Returns:
        str: The cleaned up string representation of the number.
    """
    return re.sub( '\.0+E\+0+', '', s)



#------------------------------------------------------------------------------#

#########################
# Tensor de/normalizing #
#########################

def normalize(data, m_sd = None):
    """
    Normalize the data using mean and standard deviation.

    This function normalizes the input data using the mean and standard deviation. If the `m_sd` parameter is not provided, the function computes the mean and standard deviation of the data and performs normalization. If `m_sd` is provided, it assumes that the mean and standard deviation have already been computed and uses them for normalization.

    Args:
        data (numpy.ndarray): The input data to be normalized.
        m_sd (tuple): A tuple containing the mean and standard deviation. If not provided, the mean and standard deviation will be computed from the data.

    Returns:
        numpy.ndarray: The normalized data.
    """
    if type(m_sd) == type(None):
        m = data.mean(axis = 0)
        sd = data.std(axis = 0)
        sd[np.where(sd == 0)] = 1
        return (data - m)/sd, m, sd
    else:
        m_sd[1][np.where(m_sd[1] == 0)] = 1
        return (data - m_sd[0])/m_sd[1]
        
    
def denormalize(data, train_mean, train_sd):
    """
    Denormalize the data using the mean and standard deviation.

    This function denormalizes the input data using the provided mean and standard deviation. It reverses the normalization process and brings the data back to its original scale.

    Args:
        data (numpy.ndarray): The normalized data to be denormalized.
        train_mean (numpy.ndarray): The mean used for normalization.
        train_sd (numpy.ndarray): The standard deviation used for normalization.

    Returns:
        numpy.ndarray: The denormalized data.
    """
    return data * train_sd + train_mean

#------------------------------------------------------------------------------#

######################
# phyddle print info #
######################

def phyddle_str(s, style=1, fg=34):
    """
    Apply styling to a string using ANSI escape sequences.
    
    Args:
        s (str): The string to be styled.
        style (int, optional): The style code for the text. Default is 1.
        fg (int, optional): The color code for the foreground. Default is 34 (blue).
    
    Returns:
        str: The styled string.
    """
    CSTART = f'\x1b[{style};{fg};m'
    CEND   = '\x1b[0m'
    x      = CSTART + s + CEND
    return x

def print_str(s, style=1, fg=34, verbose=True):
    """
    Prints the formatted string to the standard output.

    Parameters:
    s (str): The string to be printed.
    style (int, optional): The style of the string. Defaults to 1.
    fg (int, optional): The foreground color of the string. Defaults to 34.
    verbose (bool, optional): If True, prints the formatted string. Defaults to True.
    """
    if verbose:
        print(phyddle_str(s, style, fg))

def phyddle_header(s, style=1, fg=34):
    """
    Generate a header string for phyddle.
    
    Args:
        s (str): The type of header to generate ('title' or a step symbol).
        style (int, optional): The style code for the text. Default is 1.
        fg (int, optional): The color code for the foreground. Default is 34 (blue).
    
    Returns:
        str: The header string.
    """
    version = 'v.0.0.5'.rjust(8, ' ')
    steps = { 'sim' : 'Simulating',
              'fmt' : 'Formatting',
              'lrn' : 'Training',
              'est' : 'Estimating',
              'plt' : 'Plotting' }

    if s == 'title':
        x  = phyddle_str( '┏━━━━━━━━━━━━━━━━━━━━━━┓', style, fg ) + '\n'
        x += phyddle_str(f'┃   phyddle {version}   ┃', style, fg ) + '\n'
        x += phyddle_str( '┣━━━━━━━━━━━━━━━━━━━━━━┫', style, fg )
    
    elif s in list(steps.keys()):
        step_name = steps[s] + '...'
        step_name = step_name.ljust(13, ' ')
        x  = phyddle_str(  '┃                      ┃', style, fg ) + '\n'
        x += phyddle_str( f'┗━┳━▪ {step_name} ▪━━┛', style, fg )

    return x

def print_step_header(step, in_dir, out_dir, style=1, fg=34, verbose=True):
    """
    Generate the information string for phyddle.
    
    Args:
        step (str): The step symbol.
        in_dir (list): A list of input directories.
        out_dir (str): The output directory.
        style (int, optional): The style code for the text. Default is 1.
        fg (int, optional): The color code for the foreground. Default is 34 (blue).
       
    Returns:
        str: The information string.
    """
    # header
    run_info  = phyddle_header( step ) + '\n'
    
    # in paths
    if in_dir is not None:
        run_info += phyddle_str('  ┃')  + '\n'
        for i,_in_dir in enumerate(in_dir):
            in_path = f'{_in_dir}'
            if i == 0:
                run_info += phyddle_str(f'  ┣━━━▪ input:  {in_path}', style, fg ) + '\n'
            else:
                run_info += phyddle_str(f'  ┃             {in_path}', style, fg ) + '\n'
    
    # out path
    if out_dir is not None:
        run_info += phyddle_str('  ┃')  + '\n'
        out_path = f'{out_dir}'
    run_info += phyddle_str(f'  ┗━━━▪ output: {out_path}' ) + '\n'
    
    # print if verbose is True
    if verbose:
        print(run_info)

    return
    # return
    #return run_info

#------------------------------------------------------------------------------#





# cat .git/HEAD
# ref: refs/heads/development
# cat .git/refs/heads/development
# ef56245e012ff547c803e8a0308e6bff2718762c

class Logger:

    def __init__(self, args):
        """ 
        :class:Logger is a class that manages logging functionality for a
        project. It collects various information such as command arguments,
        package versions, system settings, and saves them into log files.
        """
        # collect info from args        
        self.args        = args
        self.arg_str     = self.make_arg_str()
        self.job_id      = self.args['job_id']
        self.log_dir     = self.args['log_dir']
        self.proj        = self.args['proj']

        # collect other info and set constants
        self.pkg_name    = 'phyddle'
        self.version     = pkg_resources.get_distribution(self.pkg_name).version
        self.commit      = '(to be done)'
        self.command     = ' '.join(sys.argv)
        self.date_obj    = datetime.now()
        self.date_str    = self.date_obj.strftime("%y%m%d_%H%M%S")
        self.max_lines   = 1e5

        # filesystem
        self.base_fn     = f'{self.pkg_name}_{self.version}_{self.date_str}_{self.job_id}'
        self.base_dir    = f'{self.log_dir}/{self.proj}'
        self.base_fp     = f'{self.base_dir}/{self.base_fn}' 
        self.fn_dict    = {
            'run' : f'{self.base_fp}.run.log',
            'simulate' : f'{self.base_fp}.simulate.log',
            'format' : f'{self.base_fp}.format.log',
            'train' : f'{self.base_fp}.train.log',
            'estimate' : f'{self.base_fp}.estimate.log',
            'plot' : f'{self.base_fp}.plot.log'
        }

        self.save_run_log()
        
        return

    def make_arg_str(self):
        """
        Creates a string representation of command arguments.

        Returns:
            str: String representation of command arguments.
        """
        ignore_keys = ['job_id']
        s = ''
        for k,v in self.args.items():
            if k not in ignore_keys:
                s += f'{k} : {v}\n'
        return s

    def save_log(self, step):
        """
        Saves log file for a specific step.

        Args:
            step (str): Step identifier.
        """
        if step == 'run':
            self.save_run_log()
        return

    def write_log(self, step, msg):
        """
        Writes a log message to a file.

        Args:
            step (str): Step identifier.
            msg (str): Log message.
        """
        fn = self.fn_dict[step]
        with open(fn, 'a') as file:
            file.write( f'{msg}\n' )
        return
    
    def save_run_log(self):
        """
        Saves run log file.
        """
        fn    = self.fn_dict['run']
        s_sys = self.make_system_log()
        s_run = self.make_phyddle_log()
        
        os.makedirs(self.base_dir, exist_ok=True)

        f = open(fn, 'w')
        f.write(s_run + '\n')
        f.write(s_sys + '\n')
        f.close()

        return
    
    def make_phyddle_log(self):
        """
        Creates a string representation of Phyddle settings.

        Returns:
            str: String representation of Phyddle settings.
        """
        s = '# PHYDDLE SETTINGS\n'
        s += f'job_id = {self.job_id}\n'
        s += f'version = {self.version}\n'
        s +=  'commit = TBD\n'
        s += f'date = {self.date_str}\n'
        s += f'command = {self.command}\n'
        s += self.make_arg_str()
        return s
    
    def make_system_log(self):
        """
        Creates a string representation of system settings.
        Note: doesn't do perfect job of finding all custom packages

        Returns:
            str: String representation of system settings.
        """

        # make dict of installed, imported packages
        d = {}
        installed_packages = { d.key for d in pkg_resources.working_set }
        for name, mod in sorted(sys.modules.items()):
            if name in installed_packages:
                if hasattr(mod, '__version__'):
                    d[name] = mod.__version__
                else:
                    try:
                        d[name] = pkg_resources.get_distribution(name).version
                    except Exception:
                        pass

        # convert into string
        s = '# SYSTEM SETTINGS\n'
        s += f'operating system = {platform.platform()}\n'
        s += f'machine architecture = {platform.machine()}\n'
        s += f'Python version = {platform.python_version()}\n'
        s +=  'Python packages:\n'
        for k,v in d.items():
            s += f'  {k} = {v}\n'
        
        return s