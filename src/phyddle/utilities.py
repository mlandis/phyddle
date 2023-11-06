#!/usr/bin/env python
"""
utilities
=========
Defines miscellaneous helper functions phyddle uses for pipeline steps.
Functions include argument parsing and checking, file conversion, managing
log files, and printing to screen.

Authors:   Michael Landis and Ammon Thompson
Copyright: (c) 2022-2023, Michael Landis and Ammon Thompson
License:   MIT
"""

# standard packages
import argparse
#import copy
import os
import pkg_resources
import platform
import re
import sys
import time
from datetime import datetime

# external packages
import dendropy as dp
import numpy as np
import pandas as pd

# phyddle imports
import __main__ as main
from . import PHYDDLE_VERSION, CONFIG_DEFAULT_FN

# Precision settings
NUM_DIGITS = 10
np.set_printoptions(floatmode='maxprec', precision=NUM_DIGITS)
pd.set_option('display.precision', NUM_DIGITS)
pd.set_option('display.float_format', lambda x: f'{x:,.3f}')

# Tensorflow info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# run mode
INTERACTIVE_SESSION = not hasattr(main, '__file__')

#------------------------------------------------------------------------------#

###################
# CONFIG LOADER   #
###################

def make_step_args(step, args):
    """Collect arguments for a step.

    This function loads the settings registry, then filters out all settings
    matching a valid step code. The returned dictionary can then be used
    to initialize the phyddle objects that execute the specified step.

    Returns:
        dict: args to initialize a phyddle step object

    """
    
    # return variable
    ret = {}

    # check that step code is valid
    if step not in 'SFTEP':
        raise ValueError
    
    # search through all registered phyddle settings
    settings = settings_registry()
    for k,v in settings.items():
        # does this setting apply to the step?
        if step in v['step']:
            # get the setting from args to return
            ret[k] = args[k]
    
    # project directories
    for p in ['sim','fmt','trn','est','plt']:
        k = f'{p}_proj'
        ret[k] = args[k]
    
    # return args the match settings for step
    return ret

def settings_registry():
    """Make registry of phyddle settings.

    This function manages all allowed phyddle settings with a dictionary. Each
    key is the name of a setting, and the value is a dictionary of that the
    properties for that setting. The properties for each setting are:
        - step : which step(s) this setting will be applied to
        - type : argument type expected by argparse
        - section: what part of the config file to appear in
        - help : argument description for argparse and config file [TBD]
        - opt  : short single-dash code for argparse (e.g. '-c' )
        - bool : default bool value for T/F strings

    Returns:
        dict : all valid phyddle settings with extra info
    """
    settings = {
        # basic phyddle options
        'cfg'              : { 'step':'',      'type':str,  'section':'Basic', 'default':'config.py',  'help':'Config file name', 'opt':'c' },
        'proj'             : { 'step':'SFTEP', 'type':str,  'section':'Basic', 'default':'my_project', 'help':'Project name(s) for pipeline step(s)', 'opt':'p' },
        # 'name'             : { 'step':'SFTEP', 'type':str,  'section':'Basic', 'default':'',           'help':'Nickname for file-set within project', 'opt':'n' },
        'step'             : { 'step':'SFTEP', 'type':str,  'section':'Basic', 'default':'SFTEP',      'help':'Pipeline step(s) defined with (S)imulate, (F)ormat, (T)rain, (E)stimate, (P)lot, or (A)ll', 'opt':'s' },
        'verbose'          : { 'step':'SFTEP', 'type':str,  'section':'Basic', 'default':'T',          'help':'Verbose output to screen?', 'bool':True, 'opt':'v' },
        'force'            : { 'step':'',      'type':None, 'section':'Basic', 'default':None,         'help':'Arguments override config file settings', 'opt':'f' },
        'make_cfg'         : { 'step':'',      'type':None, 'section':'Basic', 'default':None,         'help':"Write default config file to '__config_default.py'?'" },

        # analysis options 
        'use_parallel'     : { 'step':'SF', 'type':str, 'section':'Analysis', 'default':'T', 'help':'Use parallelization? (recommended)', 'bool':True },
        'num_proc'         : { 'step':'SF', 'type':int, 'section':'Analysis', 'default':-2, 'help':'Number of cores for multiprocessing (-N for all but N)' },
        
        # directories
        'sim_dir'          : { 'step':'SF',    'type':str, 'section':'Workspace', 'default':'../workspace/simulate', 'help':'Directory for raw simulated data' },
        'fmt_dir'          : { 'step':'FTEP',  'type':str, 'section':'Workspace', 'default':'../workspace/format',   'help':'Directory for tensor-formatted simulated data' },
        'trn_dir'          : { 'step':'FTEP',  'type':str, 'section':'Workspace', 'default':'../workspace/train',    'help':'Directory for trained networks and training output' },
        'est_dir'          : { 'step':'TEP',   'type':str, 'section':'Workspace', 'default':'../workspace/estimate', 'help':'Directory for new datasets and estimates' },
        'plt_dir'          : { 'step':'P',     'type':str, 'section':'Workspace', 'default':'../workspace/plot',     'help':'Directory for plotted results' },
        'log_dir'          : { 'step':'SFTEP', 'type':str, 'section':'Workspace', 'default':'../workspace/log',      'help':'Directory for logs of analysis metadata' },

        # simulation options
        'sim_command'      : { 'step':'S',  'type':str, 'section':'Simulate', 'default':None,    'help':'Simulation command to run single job (see documentation)' },
        'sim_logging'      : { 'step':'S',  'type':str, 'section':'Simulate', 'default':'clean', 'help':'Simulation logging style', 'choices':['clean', 'compress', 'verbose'] },
        'start_idx'        : { 'step':'SF', 'type':int, 'section':'Simulate', 'default':0,       'help':'Start replicate index for simulated training dataset' },
        'end_idx'          : { 'step':'SF', 'type':int, 'section':'Simulate', 'default':1000,    'help':'End replicate index for simulated training dataset' },
        'sim_more'         : { 'step':'S',  'type':int, 'section':'Simulate', 'default':0,       'help':'Add more simulations with auto-generated indices' },
        'sim_batch_size'   : { 'step':'S',  'type':int, 'section':'Simulate', 'default':1,       'help':'Number of replicates per simulation command' },

        # formatting options
        'encode_all_sim'   : { 'step':'F',    'type':str,  'section':'Format', 'default':'T',             'help':'Encode all simulated replicates into tensor?', 'bool':True },
        'num_char'         : { 'step':'FTE',  'type':int,  'section':'Format', 'default':None,           'help':'Number of characters' },
        'num_states'       : { 'step':'FTE',  'type':int,  'section':'Format', 'default':None,           'help':'Number of states per character' },
        'min_num_taxa'     : { 'step':'F',    'type':int,  'section':'Format', 'default':10,             'help':'Minimum number of taxa allowed when formatting' },
        'max_num_taxa'     : { 'step':'F',    'type':int,  'section':'Format', 'default':1000,           'help':'Maximum number of taxa allowed when formatting' },
        'downsample_taxa'  : { 'step':'FTE',  'type':str,  'section':'Format', 'default':'uniform',      'help':'Downsampling strategy taxon count',            'choices':['uniform'] },
        'tree_width'       : { 'step':'FTEP', 'type':int,  'section':'Format', 'default':500,            'help':'Width of phylo-state tensor' },
        #'tree_width_cats'  : { 'step':'F',   'type':int, 'section':'Format', 'default':[200, 500],     'help':'The phylo-state tensor widths for formatting training datasets (space-delimited)' },
        'tree_encode'      : { 'step':'FTE',  'type':str,  'section':'Format', 'default':'extant',       'help':'Encoding strategy for tree',                   'choices':['extant', 'serial'] },
        'brlen_encode'     : { 'step':'FTE',  'type':str,  'section':'Format', 'default':'height_brlen', 'help':'Encoding strategy for branch lengths',         'choices':['height_only', 'height_brlen'] },
        'char_encode'      : { 'step':'FTE',  'type':str,  'section':'Format', 'default':'one_hot',      'help':'Encoding strategy for character data',         'choices':['one_hot', 'integer', 'numeric'] },
        'param_est'        : { 'step':'FTE',  'type':list, 'section':'Format', 'default':None,           'help':'Model parameters to estimate' },
        'param_data'       : { 'step':'FTE',  'type':list, 'section':'Format', 'default':None,           'help':'Model parameters treated as data' },
        'char_format'      : { 'step':'FTE',  'type':str,  'section':'Format', 'default':'nexus',        'help':'File format for character data',               'choices':['csv', 'nexus'] },
        'tensor_format'    : { 'step':'FTEP', 'type':str,  'section':'Format', 'default':'hdf5',         'help':'File format for training example tensors',     'choices':['csv', 'hdf5'] },
        'save_phyenc_csv'  : { 'step':'F',    'type':str,  'section':'Format', 'default':'F',            'help':'Save encoded phylogenetic tensor encoding to csv?', 'bool':True },
        
        # training options
        'trn_objective'    : { 'step':'T',   'type':str,   'section':'Train', 'default':'param_est',   'help':'Objective of training procedure', 'choices':['param_est'] },
        'num_epochs'       : { 'step':'TEP', 'type':int,   'section':'Train', 'default':20,            'help':'Number of training epochs' },
        'trn_batch_size'   : { 'step':'TEP', 'type':int,   'section':'Train', 'default':128,           'help':'Training batch sizes' },
        'prop_test'        : { 'step':'FT',  'type':float, 'section':'Train', 'default':0.05,          'help':'Proportion of data used as test examples (assess trained network performance)' },
        'prop_val'         : { 'step':'T',   'type':float, 'section':'Train', 'default':0.05,          'help':'Proportion of data used as validation examples (diagnose network overtraining)' },
        'prop_cal'         : { 'step':'T',   'type':float, 'section':'Train', 'default':0.20,          'help':'Proportion of data used as calibration examples (calibrate CPIs)' },
        # 'combine_test_val' : { 'step':'T',   'type':bool,  'section':'Train', 'default':True,          'help':'Combine test and validation datasets when assessing network fit?' },
        'cpi_coverage'     : { 'step':'T',   'type':float, 'section':'Train', 'default':0.95,          'help':'Expected coverage percent for calibrated prediction intervals (CPIs)' },
        'cpi_asymmetric'   : { 'step':'T',   'type':str,   'section':'Train', 'default':'T',           'help':'Use asymmetric (True) or symmetric (False) adjustments for CPIs?', 'bool':True },
        'loss'             : { 'step':'T',   'type':str,   'section':'Train', 'default':'mse',         'help':'Loss function for optimization', 'choices':['mse', 'mae']},
        'optimizer'        : { 'step':'T',   'type':str,   'section':'Train', 'default':'adam',        'help':'Method used for optimizing neural network', 'choices':['adam'] },
        'metrics'          : { 'step':'T',   'type':list,  'section':'Train', 'default':['mae','acc'], 'help':'Recorded training metrics' },
        
        # estimating options
        'est_prefix'       : { 'step':'EP', 'type':str, 'section':'Estimate', 'default':None, 'help':'Predict results for this dataset' },

        # plotting options
        'plot_train_color' : { 'step':'P',  'type':str, 'section':'Plot', 'default':'blue',   'help':'Plotting color for training data elements' },
        'plot_label_color' : { 'step':'P',  'type':str, 'section':'Plot', 'default':'purple', 'help':'Plotting color for training label elements' },
        'plot_test_color'  : { 'step':'P',  'type':str, 'section':'Plot', 'default':'red',    'help':'Plotting color for test data elements' },
        'plot_val_color'   : { 'step':'P',  'type':str, 'section':'Plot', 'default':'green',  'help':'Plotting color for validation data elements' },
        'plot_aux_color'   : { 'step':'P',  'type':str, 'section':'Plot', 'default':'orange', 'help':'Plotting color for auxiliary data elements' },
        'plot_est_color'   : { 'step':'P',  'type':str, 'section':'Plot', 'default':'black',  'help':'Plotting color for new estimation elements' }
    }
    return settings


def load_config(config_fn,
                arg_overwrite=True,
                args=None):
    """Makes a dictionary of phyddle arguments.
    
    This function populates a dictionary for all phyddle settings that are
    intended to be customizable by users. The combines information from three
    main sources: the settings registry, a configuration file, and command
    line arguments. Overview of steps in function:
        1. Construct a blank argument parser with argparse
        2. Retrieve dictionary of registered settings
        3. Translate registered settings into argparse arguments
        4. Load configuration file
        5. Convert configuration dictionary into settings
        6. Apply command-line argument as settings
        7. Validate settings (when possible)
        8. Return settings

    Arguments:
        config_fn (str): The config file name.
        arg_overwrite (bool): Overwrite config file arguments?
        args (list): List of provided arguments (mainly for debugging)

    Returns:
        dict: The loaded configuration.

    """
    
    # use command line sys.argv if no args provided
    if INTERACTIVE_SESSION:
        args = []
    elif args is None:
        args = sys.argv[1:]

    # argument parsing
    desc = 'Software to fiddle around with deep learning for phylogenetic models'
    parser = argparse.ArgumentParser(description=desc)
    #, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # read settings registry and populate argument parser
    settings = settings_registry()
    for k,v in settings.items():
        arg_opt = []
        if 'opt' in v:
            arg_opt.append(f'-{v["opt"]}')
        arg_opt.append(f'--{k}')
        arg_help = v['help']
        arg_type = v['type']
        
        if arg_type is not None:
            if 'choices' in v:
                arg_choices = v['choices']
                parser.add_argument(*arg_opt, dest=k, type=arg_type,
                                    choices=arg_choices, help=arg_help,
                                    metavar='')
            else:
                parser.add_argument(*arg_opt, dest=k, type=arg_type,
                                    help=arg_help, metavar='')
        else:
             parser.add_argument(*arg_opt, action='store_true', help=arg_help)
   
    # COMMAND LINE SETTINGS
    args = parser.parse_args(args)

    # DEFAULT CONFIG FILE SETTINGS
    # make/overwrite default config, if requested
    if args.make_cfg:
        make_config_fn = CONFIG_DEFAULT_FN
        if arg_overwrite and args.cfg is not None:
            make_config_fn = args.cfg
        make_default_config(make_config_fn)
        print_str(f"Created default config as '{make_config_fn}' ...")
        sys.exit()
    if not os.path.exists(CONFIG_DEFAULT_FN):
        msg = f"Default config file '{CONFIG_DEFAULT_FN} not found. Creating "
        msg += "default config file in current directory."
        print_warn(msg)
        make_default_config(CONFIG_DEFAULT_FN)
    # load config into namespace
    namespace = {}        
    with open(CONFIG_DEFAULT_FN) as file:
        code = file.read()
        exec(code, namespace)
    # move imported args into local variable
    default_args = namespace['args']
    
    # PROJECT CONFIG FILE SETTINGS
    if arg_overwrite and args.cfg is not None:
        config_fn = args.cfg
    if not os.path.exists(config_fn):
        msg = (f"Project config file '{config_fn}' not found. If you have a "
                "project config file, please verify access to the file. There "
                "are two easy ways to create a user config file: (1) copy the "
               f"default config file '{CONFIG_DEFAULT_FN}' and modify it as "
                "needed, or (2) download an existing project config file from "
                "https://github.com/mlandis/phyddle/tree/main/scripts." )
        print_err(msg)
        sys.exit()

    namespace = {}
    with open(config_fn) as file:
        code = file.read()
        exec(code, namespace)

    # move imported args into local variable
    file_args = namespace['args']
    
    # MERGE SETTINGS
    # merge default, user_file, and user_cmd settings
    for k in settings.keys():
        m = reconcile_settings(default_args, file_args, args, k)

    # fix convert string-valued bool to true bool
    m = fix_arg_bool(m)

    # update steps
    if m['step'] == 'A':
        m['step'] = 'SFTEP'
    
    # check arguments are valid
    check_args(m)

    # set steps & projects
    m = add_step_proj(m)
    
    # add session info
    date_obj = datetime.now()
    m['date'] = date_obj.strftime("%y%m%d_%H%M%S")
    m['job_id'] = generate_random_hex_string(7)
    
    # print header?
    verbose = m['verbose']
    if verbose:
        print(phyddle_header('title'))

    # return new args
    return m

def fix_arg_bool(m):
    """Convert bool-str arguments to True/False bool."""
    settings = settings_registry()
    for k,v in settings.items():
        if 'bool' in v and type(m[k]) != str:
            print(m[k],type(m[k]))
            raise Exception(f"Invalid argument: {k} must be a string")
        elif 'bool' in v and type(m[k]) == str:
            arg_val = m[k]
            arg_val_new = str2bool(arg_val)
            if arg_val_new is None:
                raise ValueError(f'{arg_val} invalid value for {k}')
            m[k] = arg_val_new
    return m

def str2bool(x):
    """Convert a str value to True/False"""
    if x.lower() in ['true', 'yes', 't', 'y', '1' ]:
        return True
    elif x.lower() in ['false', 'no', 'f', 'n', '0' ]:
        return False
    else:
        return None

def check_args(args):
    """Checks if the given arguments meet certain conditions.

    Parameters:
    args (dict): A dictionary containing the arguments.

    Raises:
    AssertionError: If any of the conditions are not met.
    """

    # probably should turn all of these into helpful error messages, with
    # links to user facing documentation

    # string values
    assert all([s in 'ASFTEP' for s in args['step']])
    assert args['sim_logging']       in ['clean', 'verbose', 'compress']
    assert args['tree_encode']       in ['serial', 'extant']
    assert args['brlen_encode']      in ['height_only', 'height_brlen']
    assert args['char_encode']       in ['one_hot', 'integer', 'numeric']
    assert args['tensor_format']     in ['csv', 'hdf5']
    assert args['char_format']       in ['csv', 'nexus']
    assert args['trn_objective']     in ['param_est', 'model_test']
    
    # numerical values
    assert args['start_idx'] >= 0
    assert args['end_idx'] >= 0
    assert args['start_idx'] <= args['end_idx']
    assert args['sim_more'] >= 0
    assert args['min_num_taxa'] >= 0
    assert args['max_num_taxa'] >= 0
    assert args['min_num_taxa'] <= args['max_num_taxa']
    assert args['num_states'] > 0
    assert args['num_char'] > 0
    assert args['num_epochs'] > 0
    assert args['trn_batch_size'] > 0
    assert args['sim_batch_size'] > 0
    assert args['cpi_coverage'] >= 0. and args['cpi_coverage'] <= 1.
    assert args['prop_test'] >= 0. and args['prop_test'] <= 1.
    assert args['prop_val'] >= 0. and args['prop_val'] <= 1.
    assert args['prop_cal'] >= 0. and args['prop_cal'] <= 1.

    # done
    return

def add_step_proj(args): #steps, proj):
    """Manages project directories for steps.

    This function determines which step will be used from args. Next it
    processes the project string to determine the project(s) name(s) across
    all steps. Last, it creates step-specific project directories and stores
    them back into args.

    Args:
        args (dict): Un-updated phyddle settings dictionary
    
    Returns:
        dict: Updated the phyddle settings dictionary.

    """
    # get relevant args
    steps = args['step']
    proj = args['proj']
    
    # different ways of naming steps
    d_map = { 'S': ('sim', 'simulate'),
              'F': ('fmt', 'format'),
              'T': ('trn', 'train'),
              'E': ('est', 'estimate'),
              'P': ('plt', 'plot'),
              'L': ('log', 'log') }
    
    # parse input string
    d_toks = {}
    proj_toks = proj.split(',')
    for p in proj_toks:
        if ':' not in p:
            d_toks['A'] = p
        else:
            k,v = p.split(':')
            d_toks[k] = v

    # handle all-step ('A') first
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
            msg = f"Step {s} ({d_map[s][1]}) has no assigned project name"
            raise ValueError(msg)

    for k in d_arg.keys():
        k_str = k + '_proj'
        args[k_str] = d_arg[k]

    return args

def make_default_config(config_fn):
    """Generate default config file.

    Processes all items in the settings_registry to create a formatted
    default config dictionary. Writes config to file.

    """
    
    # get settings registry
    settings = settings_registry()
    
    # sort settings by section
    section_settings = {}
    for k,v in settings.items():
        sect = v['section']
        if sect not in section_settings.keys():
            section_settings[sect] = {}
        section_settings[sect][k] = v
    
    # constant tokens
    s_assign = ' : '
    s_comment = '  # '
    s_indent = '  '
    
    # token lengths               # token examples
    len_indent = len(s_indent)    # "  "
    len_assign = len(s_assign)    # " : "
    len_comment = len(s_comment)  # " # ""
    len_punct = len_indent + len_assign + len_comment
    len_key = 20                  # "'num_epochs'".ljust(len_key, ' ')
    len_value = 20                # str(20).ljust(len_value, ' ')
    len_help = 32                 # "Number of training examples"
    len_line = 80
    
    section_str = {}
    for i,(k1,v1) in enumerate(section_settings.items()):
        # section header
        s_sect  =  "  #-------------------------------#\n"
        s_sect +=  "  # " + str(k1).ljust(30, ' ') +      "#\n"
        s_sect +=  "  #-------------------------------#\n"

        # max widths for sections
        max_key, max_value, max_help = 0, 0, 0

        # info for sections
        v_key, v_value, v_help = [], [], []

        # get key,value,help and max sizes per section
        for k2,v2 in v1.items():
            # key
            s_key = f"'{str(k2)}'" #.ljust(len_key, ' ')
            v_key.append(s_key)
            max_key = max(max_key, len(s_key))

            # value
            s_value = str(v2['default'])
            if (v2['type'] is str and s_value != 'None'):
                s_value = "'" + s_value + "'"
            v_value.append(s_value+',')
            max_value = max(max_value, len(s_value))

            # help
            s_help = v2['help']
            v_help.append(s_help)
            max_help = max(max_help, len(s_help))

        # make key,value,help strings for each section entry
        for ki,vi,hi in zip(v_key, v_value, v_help):
            
            width_key = max(len_key, max_key)
            width_value = max(len_value, max_value)
            width_help = len_line - (width_key + width_value + len_punct)
            
            s_tok = []
            s_tok.append(s_indent)
            s_tok.append(ki.ljust(width_key, ' '))
            s_tok.append(s_assign)
            s_tok.append(vi.ljust(width_value, ' '))
            s_tok.append(s_comment)
            s_tok.append(hi.ljust(width_help, ' '))
        
            s_sect += ''.join(s_tok) + '\n'

        section_str[k1] = s_sect

    # build file content
    s_cfg  = "#==============================================================================#\n"
    s_cfg += "# Default phyddle config file                                                  #\n"
    s_cfg += "#==============================================================================#\n"
    s_cfg += "\n"
    s_cfg += "args = {\n"
    for k,v in section_str.items():
        s_cfg += v + '\n'
    s_cfg += "}\n"

    # write file content
    f = open(config_fn, 'w')
    f.write(s_cfg)
    f.close()
    
    # done
    return

# update arguments from defaults, when provided
def reconcile_settings(default_args, file_args, cmd_args, var):
        """Reconciles settings from all sources.

        Settings are applied and overwritten in this order:
        1. default settings
        2. config file settings
        3. command line settings

        Settings are applied to default settings before being returned.

        Args:
            default_args (dict): Default settings
            file_args (dict): Config file settings
            cmd_args (dict): Command line settings
            var (str): Key for setting to be updated
        
        Returns:
            default_args (dict): Updated default settings.

        """
        
        # first apply file args
        if var in file_args.keys():
            x_file = file_args[var]
            if x_file is not None:
                default_args[var] = x_file

        # then apply command args
        x_cmd = getattr(cmd_args, var)
        if x_cmd is not None:
            default_args[var] = x_cmd

        return default_args


def strip_py(s):
    """Remove .py suffix from file."""
    return re.sub(r'\.py$', '', s)

def generate_random_hex_string(length):
    """
    Generates a random hex string of a given length. Used for phyddle run id.

    Args:
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

def set_seed(seed_value):
    
    # see: https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
    
    import tensorflow as tf
    import numpy as np
    import random
    import os
    
    os.environ['PYTHONHASHSEED']=str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

    return

def get_time():
    """Get current clock time."""
    t = time.localtime()
    s = time.strftime("%H:%M:%S", t)
    return time.mktime(t),s

def get_time_str():
    """Make time string."""
    return get_time()[1]

def get_time_diff(start_time, end_time):
    """Make time-difference string.
    
    Args:
        start_time (object): start time
        end_time (object): end time

    Returns:
        str: String reports time difference, HH:MM:SS
    """
    difference_seconds = end_time - start_time
    hours, remainder = divmod(difference_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    h = f'{int(hours)}'.rjust(2, '0')
    m = f'{int(minutes)}'.rjust(2, '0')
    s = f'{int(seconds)}'.rjust(2, '0')
    ret = f'{h}:{m}:{s}'
    return ret


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

def get_num_tree_row(tree_encode, brlen_encode):
    """Gets number of tree rows.
    
    Computes number of rows for encoding tree information using a compact
    phylogenetic vector + states (CPV+S) format.

    CBLV (2 rows) is used if tree_encode == 'serial'. CDV (1 row) is used if
    tree_encode == 'extant'.

    No extra rows are used if brlen_encode == 'height_only'. Two extra rows
    are used if brlen_encode == 'height_brlen'.

    Args:
        tree_encode (str): Use CBLV (serial) or CDV (extant) for tree encoding.
        brlen_encode (str): Use height-only or height + brlen encoding.

    Returns:
        int: The number of rows for the CPV encoding.

    """

    if tree_encode == 'serial':
        num_tree_row = 2
    elif tree_encode == 'extant':
        num_tree_row = 1

    if brlen_encode == 'height_only':
        num_tree_row += 0
    elif brlen_encode == 'height_brlen':
        num_tree_row += 2

    return num_tree_row

def get_num_char_row(state_encode_type, num_char, num_states):
    """Gets number of character rows.
    
    Computes number of rows for encoding state information using a compact
    phylogenetic vector + states (CPV+S) format.

    Integer encoding uses one row per character, with any number of states
    per character.

    One-hot encoding uses k rows per character, where k is the number of
    states.

    Args:
        char_encode (str): Use integer or one_hot encoding
        
    Returns:
        int: The number of rows for the +S encoding.
    
    """
    
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
        fn (str): The path of the file to write the string to.

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
        dp.Tree: The parsed phylogenetic tree object, or None if
                 the tree cannot be read.

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

def convert_csv_to_array(dat_fn, char_encode, num_states=None):
    """Converts CSV to array format."""
    if char_encode == 'numeric' or char_encode == 'integer':
        dat = convert_csv_to_numeric_array(dat_fn, num_states)
    elif char_encode == 'one_hot':
        dat = convert_csv_to_onehot_array(dat_fn, num_states)
    else:
        return NotImplementedError
    return dat



def convert_csv_to_numeric_array(dat_fn, pca_compress=None):
    """Converts a csv file to an integer-encoded pandas DataFrame."""
    # read as pandas
    dat = pd.read_csv(dat_fn, delimiter=',', index_col=0, header=None).T 

    # return
    return dat

def convert_csv_to_onehot_array(dat_fn, num_states):
    """Converts a csv file to an integer-encoded pandas DataFrame."""
    
    # read data
    dat_raw = pd.read_csv(dat_fn, delimiter=',', index_col=0, header=None).T

    # get num taxa (columns)
    num_taxa = dat_raw.shape[1]

    # check/unify number of state per row
    print(type(num_states))
    if type(num_states) is int:
        num_states = [ num_states ] * dat_raw.shape[0]

    assert(dat_raw.shape[0] == len(num_states))
    assert(all([type(i) is int for i in num_states]))

    # make csv
    num_rows = sum(num_states)
    zero_data = np.zeros(shape=(num_rows,num_taxa), dtype='int')
    dat = pd.DataFrame(zero_data, columns=dat_raw.columns)

    # do one-hot encoding
    j = 0
    for i,ns in enumerate(num_states):
        k = j + num_states[i]
        dat.iloc[j:k,:] = to_categorical(dat_raw.iloc[i,:], num_classes=ns, dtype='int').T
        j = k
    
    # done!
    return dat

def to_categorical(y, num_classes=None, dtype="float32"):
    """Converts a class vector (integers) to binary class matrix.

    Importing from keras is very slow (multiple seconds), so this function
    is a direct copy & paste keras/utils/np_utils.py
    
    E.g. for use with `categorical_crossentropy`.

    Args:
        y: Array-like with class values to be converted into a matrix
            (integers from 0 to `num_classes - 1`).
        num_classes: Total number of classes. If `None`, this would be inferred
          as `max(y) + 1`.
        dtype: The data type expected by the input. Default: `'float32'`.

    Returns:
        A binary matrix representation of the input as a NumPy array. The class
        axis is placed last.
    """
    y = np.array(y, dtype="int")
    input_shape = y.shape

    # Shrink the last dimension if the shape is (..., 1).
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])

    y = y.reshape(-1)
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

def convert_nexus_to_array(dat_fn, char_encode, num_states=None):
    """
    Convert Nexus file to array format.

    Args:
        dat_fn (str): The file name of the Nexus file.
        char_encode (str): The type of character encoding to use. 
                           Valid options are "integer" or "one_hot".
        num_states (int): The number of states. Only applicable if char_encode
                          is "one_hot".

    Returns:
    dat (array): The converted array.
    """
    if char_encode == 'integer':
        dat = convert_nexus_to_integer_array(dat_fn)
    elif char_encode == 'one_hot':
        dat = convert_nexus_to_onehot_array(dat_fn, num_states)
    else:
        return NotImplementedError

    return dat

def convert_nexus_to_integer_array(dat_fn):
    """Converts a NEXUS file to an integer-encoded pandas DataFrame.

    Reads the NEXUS file specified by `dat_fn`, extracts the data matrix,
    and constructs a pandas DataFrame where rows represent character states
    and columns represent taxa.

    Args:
        dat_fn (str): The file name or path of the NEXUS file.

    Returns:
        pd.DataFrame: The pandas DataFrame representing the data matrix.

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

    Reads the NEXUS file specified by `dat_fn`, extracts the data matrix, and
    constructs a pandas DataFrame where rows represent character states and
    columns represent taxa.

    Args:
        dat_fn (str): The file name or path of the NEXUS file.
        num_states (int, optional): Number of states to one-hot encode. Learned
            from length of symbols when None. Defaults to None.

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
    Converts a phylogenetic tree in NHX format to a NEXUS file with taxon-state
    data.

    Reads the phylogenetic tree file in NHX format specified by `phy_nex_fn`
    and converts it to a NEXUS file containing taxon-state data. The binary
    state representations are based on the provided `int2vec` mapping.

    Args:
        phy_nex_fn (str): The file name or path of the phylogenetic tree file
            in NHX format.
        int2vec (List[int]): The mapping of integer states to binary state
            vectors.

    Returns:
        str: The NEXUS file content as a string.

    """

    # get num regions from size of bit vector
    num_char = len(int2vec[0])

    # get tip names and states from NHX tree
    nex_file = open(phy_nex_fn, 'r')
    nex_str  = nex_file.readlines()[3]
    # MJL add regex string qualifier
    matches  = re.findall(pattern=r'([0-9]+)\[\&type="([A-Z]+)",location="([0-9]+)"', string=nex_str)
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
    """Prunes a phylogenetic tree by removing non-extant taxa and writes the
    pruned tree to a file.

    The function takes a phylogenetic tree `phy` and a file name `prune_fn` as 
    input. It prunes the tree by removing non-extant taxa and writes the pruned
    tree to the specified file.

    Args:
        phy (Tree): The input phylogenetic tree.
        prune_fn (str): The file name or path to write the pruned tree.

    Returns:
        dp.Tree: The pruned phylogenetic tree if pruning is successful,
                 or None if the pruned tree would have fewer than two
                 leaf nodes (invalid tree).

    """
    # copy input tree
    phy_ = phy # copy.deepcopy(phy)
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


def make_downsample_phy(phy, down_fn, max_taxa, strategy):
    """Subsampling of taxa."""
    if strategy == 'uniform':
        phy = make_uniform_downsample_phy(phy, down_fn, max_taxa)
    else:
        raise NotImplementedError    
    return phy

def make_uniform_downsample_phy(phy, down_fn, max_taxa):
    """Uniform random subsampling of taxa."""
    # copy input tree
    phy_ = phy #copy.deepcopy(phy)
    
    # get number of taxa
    leaf_nodes = phy_.leaf_nodes()
    num_taxa = len(leaf_nodes)

    # if downsampling is needed
    if num_taxa > max_taxa:
        
        #drop_taxon_labels = [ str(nd.taxon).strip("'").replace(' ','_') for nd in leaf_nodes ]
        #np.random.shuffle(drop_taxon_labels)
        #drop_taxon_labels = drop_taxon_labels[max_taxa:]
        # MJL: note, Format bottleneck with large trees. It should be possible
        #      to write faster code when we don't care about taxon labels or
        #      downstream use of the dendropy object?
        # https://dendropy.org/_modules/dendropy/datamodel/treemodel/_tree#Tree.prune_taxa_with_labels
        #phy_.prune_taxa_with_labels( drop_taxon_labels )
        
        # shuffle taxa indices
        rand_idx = list(range(num_taxa))
        np.random.shuffle(rand_idx)
        
        # get all taxa beyond max_taxa threshold
        drop_taxa = []
        for i in rand_idx[max_taxa:]:
            drop_taxa.append(phy.taxon_namespace[i])

        # drop those taxa
        phy.prune_taxa(drop_taxa)

        # verify resultant tree size
        assert(len(phy.leaf_nodes()) == max_taxa)

    # save downsampled tree
    phy_.write(path=down_fn, schema='newick')

    # done
    return phy_

# make matrix with parameter values, lower-bounds, upper-bounds: 3D->2D
def make_param_VLU_mtx(A, param_names):
    """Make paramter Value-Lower-Upper matrix.

    This function takes a parameter matrix A and a list of parameter names and
    creates a pandas DataFrame with combined header indices. The resulting
    DataFrame has columns representing different statistics (value, lower,
    upper), replicated indices, and parameters. The parameter names and
    statistics are combined to form the column headers.

    Args:
        A (numpy.ndarray): The parameter matrix.
        param_names (str[]): A list of parameter names.

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
    """Convert a numpy array to a clean string representation.

    This function takes a numpy array `x` and converts it to a clean string
    representation. The resulting string is obtained by formatting the array
    with a comma separator, removing the square brackets, and replacing line
    breaks and unnecessary whitespace characters. The string representation is
    useful for displaying or saving the array in a clean and readable format.

    Args:
        x (numpy.ndarray): The numpy array to convert.

    Returns:
        str: The clean string representation of the numpy array.
    """
    s = np.array2string(x, separator=',', max_line_width=1e200,
                        threshold=1e200, edgeitems=1e200, precision=10,
                        floatmode='maxprec')
    s = re.sub(r'[\[\]]', '', string=s)
    s = re.sub(r',\n ', '\n', string=s)
    s = s + '\n'
    return s

#------------------------------------------------------------------------------#

#########################
# Tensor de/normalizing #
#########################

def normalize(data, m_sd = None):
    """
    Normalize the data using mean and standard deviation.

    This function normalizes the input data using the mean and standard
    deviation. If the `m_sd` parameter is not provided, the function computes
    the mean and standard deviation of the data and performs normalization.
    If `m_sd` is provided, it assumes that the mean and standard deviation
    have already been computed and uses them for normalization.

    Args:
        data (numpy.ndarray): The input data to be normalized.
        m_sd (tuple): A tuple containing the mean and standard deviation. If
            not provided, the mean and standard deviation will be computed from
            the data.

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

def denormalize(data, m_sd, exp=False, tol=300):
    """
    Denormalize the data using the mean and standard deviation.

    This function denormalizes the input data using the provided mean and
    standard deviation. It reverses the normalization process and brings the
    data back to its original scale.

    Args:
        data (numpy.ndarray): The normalized data to be denormalized.
        train_mean (numpy.ndarray): The mean used for normalization.
        train_sd (numpy.ndarray): The standard deviation used for normalization.

    Returns:
        numpy.ndarray: The denormalized data.
    """
    if exp:
        x = data * m_sd[1] + m_sd[0]
        x[x>=tol] = tol
        x[x<=-tol] = -tol
        return np.exp(x)
    else:
        return data * m_sd[1] + m_sd[0]

#------------------------------------------------------------------------------#

######################
# phyddle print info #
######################

def phyddle_str(s, style=1, fg=34):
    """Make printable phyddle output string.
    
    Args:
        s (str): The string to be styled.
        style (int, optional): Style code.
        fg (int, optional): Foreground color code.
    
    Returns:
        str: The styled string.

    """

    CSTART = f'\x1b[{style};{fg};m'
    CEND   = '\x1b[0m'
    x      = CSTART + s + CEND
    return x

def print_str(s, verbose=True, style=1, fg=34):
    """Prints a phyddle string to the standard output.

    Args:
        s (str): The string to be styled.
        style (int, optional): Style code.
        fg (int, optional): Foreground color code.
        verbose (bool, optional): Prints the formatted string if True.

    """

    if verbose:
        print(phyddle_str(s, style, fg))
    return

def print_err(s, verbose=True, style=1, fg=34):
    """Prints a phyddle error to the standard output.

    Args:
        s (str): The string to be styled.
        style (int, optional): Style code.
        fg (int, optional): Foreground color code.
        verbose (bool, optional): Prints the formatted string if True.

    """
    print_str(f'ERROR: {s}', verbose, style, fg)
    return

def print_warn(s, verbose=True, style=1, fg=34):
    """Prints a phyddle warning to the standard output.

    Args:
        s (str): The string to be styled.
        style (int, optional): Style code.
        fg (int, optional): Foreground color code.
        verbose (bool, optional): Prints the formatted string if True.

    """

    print_str(f'WARNING: {s}', verbose, style, fg)
    return

def phyddle_header(s, style=1, fg=34):
    """Generate a phyddle header string.
    
    Args:
        s (str): The string to be styled.
        style (int, optional): Style code.
        fg (int, optional): Foreground color code.

    Returns:
        str: The header string.

    """

    version = f'v{PHYDDLE_VERSION}'.rjust(8, ' ')

    steps = { 'sim' : 'Simulating',
              'fmt' : 'Formatting',
              'trn' : 'Training',
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

def print_step_header(step, in_dir, out_dir, verbose=True, style=1, fg=34):
    """Generate a phyddle step info string.
    
    Args:
        step (str): The step symbol.
        in_dir (list): A list of input directories.
        out_dir (str): The output directory.
        style (int, optional): Style code.
        fg (int, optional): Foreground color code.
        
    Returns:
        str: The information string.

    """

    # header
    run_info  = phyddle_header( step ) + '\n'
    
    # in paths
    plot_bar = True
    if in_dir is not None:
        run_info += phyddle_str('  ┃')  + '\n'
        plot_bar = False
        for i,_in_dir in enumerate(in_dir):
            in_path = f'{_in_dir}'
            if i == 0:
                run_info += phyddle_str(f'  ┣━━━▪ input:  {in_path}', style, fg ) + '\n'
            else:
                run_info += phyddle_str(f'  ┃             {in_path}', style, fg ) + '\n'
    
    # out path
    if out_dir is not None:
        #run_info += phyddle_str('  ┃')  + '\n'
        out_path = f'{out_dir}'
    if plot_bar:
        run_info += phyddle_str('  ┃')  + '\n'
    
    run_info += phyddle_str(f'  ┗━━━▪ output: {out_path}' ) + '\n'
    
    # print if verbose is True
    if verbose:
        print(run_info)

    return

#------------------------------------------------------------------------------#

# cat .git/HEAD
# ref: refs/heads/development
# cat .git/refs/heads/development
# ef56245e012ff547c803e8a0308e6bff2718762c

class Logger:
    """ 
    Logger  manages logging functionality for a project. It collects
    various information such as command arguments, package versions,
    system settings, and saves them into log files.
    """
    def __init__(self, args):
        
        # collect info from args        
        self.args        = args
        self.arg_str     = self.make_arg_str()
        self.job_id      = self.args['job_id']
        self.log_dir     = self.args['log_dir']
        self.date_str    = self.args['date']
        self.proj        = self.args['proj']

        # collect other info and set constants
        self.pkg_name    = 'phyddle'
        self.version     = PHYDDLE_VERSION #pkg_resources.get_distribution(self.pkg_name).version
        self.commit      = '(to be done)'
        self.command     = ' '.join(sys.argv)
        self.max_lines   = 1e5

        # filesystem
        self.base_fn     = f'{self.pkg_name}_{self.version}_{self.date_str}'
        self.base_dir    = f'{self.log_dir}/{self.proj}'
        self.base_fp     = f'{self.base_dir}/{self.base_fn}' 
        self.fn_dict    = {
            'run' : f'{self.base_fp}.run.log',
            'sim' : f'{self.base_fp}.simulate.log',
            'fmt' : f'{self.base_fp}.format.log',
            'trn' : f'{self.base_fp}.train.log',
            'est' : f'{self.base_fp}.estimate.log',
            'plt' : f'{self.base_fp}.plot.log'
        }

        self.save_run_log()
        
        return

    def make_arg_str(self):
        """Creates a string representation of command arguments.

        Returns:
            str: String representation of command arguments.

        """

        ignore_keys = ['job_id']
        s = ''
        for k,v in self.args.items():
            if k not in ignore_keys:
                s += f'{k}\t{v}\n'
        return s

    def save_log(self, step):
        """Saves log file for a specific step.

        Args:
            step (str): Step identifier.
        """

        if step == 'run':
            self.save_run_log()
        return

    def write_log(self, step, msg):
        """Writes a log message to a file.

        Args:
            step (str): Step identifier.
            msg (str): Log message.

        """

        assert(step in self.fn_dict.keys())
        fn = self.fn_dict[step]
        with open(fn, 'a') as file:
            file.write( f'{msg}\n' )
        return
    
    def save_run_log(self):
        """Saves run log file."""
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
        """Creates a string representation of phyddle settings.

        Returns:
            str: String representation of phyddle settings.

        """

        s = '# PHYDDLE SETTINGS\n'
        s += f'job_id\t{self.job_id}\n'
        s += f'version\t{self.version}\n'
        #s +=  'commit = TBD\n'
        s += f'date = {self.date_str}\n'
        s += f'command\t{self.command}\n'
        s += self.make_arg_str()
        return s
    
    def make_system_log(self):
        """Creates a string representation of system settings.
        
        Note: doesn't do perfect job of finding all custom packages. Need
        to revisit this, but not urgent.

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
        s += f'operating system\t{platform.platform()}\n'
        s += f'machine architecture\t{platform.machine()}\n'
        s += f'Python version\t{platform.python_version()}\n'
        s +=  'Python packages:\n'
        for k,v in d.items():
            s += f'{k}\t{v}\n'
        
        return s
    
#------------------------------------------------------------------------------#

# class Tensor:
#     """Batch of tensor-shaped datasets"""
    
#     def __init__(self, data, norms=None, calibs=None):
        
#         # initialize values
#         self.data = self.set_format(data)
#         self.standardizations = norms
#         self.calibs = calibs
#         self.names = self.data.columns
#         self.shape = self.data.shape

#         if norms is None:
#             self.data_norm, mean, sd = self.normalize(self.data)
#         else:
#             mean = norms[0,:]
#             sd = norms[1,:]
#             self.data_norm, mean, sd = self.normalize(self.data, mean, sd)
#         self.mean = mean
#         self.sd = sd

#         return

#     def apply_scalers(self, data):

#         return
    
#     def normalize(self, data, m_sd=None):
#         """
#         Normalize the data using mean and standard deviation.

#         This function normalizes the input data using the mean and standard deviation. If the `m_sd` parameter is not provided, the function computes the mean and standard deviation of the data and performs normalization. If `m_sd` is provided, it assumes that the mean and standard deviation have already been computed and uses them for normalization.

#         Args:
#             data (numpy.ndarray): The input data to be normalized.
#             m_sd (tuple): A tuple containing the mean and standard deviation. If not provided, the mean and standard deviation will be computed from the data.

#         Returns:
#             numpy.ndarray: The normalized data.
#         """
#         if type(m_sd) == type(None):
#             m = data.mean(axis = 0)
#             sd = data.std(axis = 0)
#             sd[np.where(sd == 0)] = 1
#             return (data - m)/sd, m, sd
#         else:
#             m_sd[1][np.where(m_sd[1] == 0)] = 1
#             return (data - m_sd[0])/m_sd[1]
            
        

#     def denormalize(self):
#         return
    
#     def calibrate(self):
#         return
    
#     def decalibrate(self):
#         return





##################
# Helper Classes #
##################

# # model events
# class Event:
#     """
#     Event objects define an event for a Poisson process with discrete-valued
#     states, such as continuous-time Markov processes. Note, that
#     phylogenetic birth-death models and SIR models fall into this class.
#     The Event class was originally designed for use with chemical
#     reaction simulations using the MASTER plugin in BEAST.
#     """
#     # initialize
#     def __init__(self, idx, r=0.0, n=None, g=None, ix=None, jx=None):
#         """
#         Create an Event object.

#         Args:
#             idx (dict): A dictionary containing the indices of the event.
#             r (float): The rate of the event.
#             n (str): The name of the event.
#             g (str): The reaction group of the event.
#             ix (list): The reaction quantities (reactants) before the event.
#             jx (list): The reaction quantities (products) after the event.
#         """
#         self.i = -1
#         self.j = -1
#         self.k = -1
#         self.idx = idx
#         if 'i' in idx:
#             self.i = idx['i']
#         if 'j' in idx:
#             self.j = idx['j']
#         if 'k' in idx:
#             self.k = idx['k']
#         self.rate = r
#         self.name = n
#         self.group = g
#         self.ix = ix
#         self.jx = jx
#         self.reaction = ' + '.join(ix) + ' -> ' + ' + '.join(jx)
#         return
        
#     # make print string
#     def make_str(self):
#         """
#         Creates a string representation of the event.

#         Returns:
#             str: The string representation of the event.
#         """
#         s = 'Event({name},{group},{rate},{idx})'.format(name=self.name, group=self.group, rate=self.rate, idx=self.idx)        
#         #s += ')'
#         return s
    
#     # representation string
#     def __repr__(self):
#         """
#         Returns the representation of the event.

#         Returns:
#             str: The representation of the event.
#         """
#         return self.make_str()
    
#     # print string
#     def __str__(self):
#         """
#         Returns the string representation of the event.

#         Returns:
#             str: The string representation of the event.
#         """
#         return self.make_str()


# # state space
# class States:
#     """
#     States objects define the state space that a model operates upon. Event
#     objects define transition rates and patterns with respect to States. The
#     central purpose of States is to manage different representations of
#     individual states in the state space, e.g. as integers, strings, vectors.
#     """
#     def __init__(self, lbl2vec):
#         """
#         Create a States object.

#         Args:
#             lbl2vec (dict): A dictionary with labels (str) as keys and vectors
#                             of states (int[]) as values.
#         """
#         # state space dictionary (input)
#         self.lbl2vec      = lbl2vec

#         # basic info
#         self.int2lbl        = list( lbl2vec.keys() )
#         self.int2vec        = list( lbl2vec.values() )
#         self.int2int        = list( range(len(self.int2vec)) )
#         self.int2set        = list( [ tuple([y for y,v in enumerate(x) if v == 1]) for x in self.int2vec ] )
#         self.lbl_one        = list( set(''.join(self.int2lbl)) )
#         self.num_char       = len( self.int2vec[0] )
#         self.num_states     = len( self.lbl_one )

#         # relational info
#         self.lbl2int = {k:v for k,v in list(zip(self.int2lbl, self.int2int))}
#         self.lbl2set = {k:v for k,v in list(zip(self.int2lbl, self.int2set))}
#         self.lbl2vec = {k:v for k,v in list(zip(self.int2lbl, self.int2vec))}
#         self.vec2int = {tuple(k):v for k,v in list(zip(self.int2vec, self.int2int))}
#         self.vec2lbl = {tuple(k):v for k,v in list(zip(self.int2vec, self.int2lbl))}
#         self.vec2set = {tuple(k):v for k,v in list(zip(self.int2vec, self.int2set))}
#         self.set2vec = {tuple(k):v for k,v in list(zip(self.int2set, self.int2vec))}
#         self.set2int = {tuple(k):v for k,v in list(zip(self.int2set, self.int2int))}
#         self.set2lbl = {tuple(k):v for k,v in list(zip(self.int2set, self.int2lbl))}
#         self.int2vecstr = [ ''.join([str(y) for y in x]) for x in self.int2vec ]
#         self.vecstr2int = { v:i for i,v in enumerate(self.int2vecstr) }
       
#         # done
#         return

#     def make_str(self):
#         """
#         Creates a string representation of the state space.

#         Returns:
#             str: The string representation of the state space.
#         """
#         # state space: {'A': [1, 0, 0], 'B': [0, 1, 0], 'C': [0, 0, 1], 'AB': [1, 1, 0], 'AC': [1, 0, 1], 'BC': [0, 1, 1], 'ABC': [1, 1, 1]}
#         # string: Statespace(A,0,100;B,1,010;C,2,001;AB,3,110;AC,4,101;BC,5,011;ABC,6,111)
#         s = 'Statespace('
#         x = []
#         for i in self.int2int:
#             # each state in the space is reported as STRING,INT,VECTOR;
#             x.append( self.int2lbl[i] + ',' + str(self.int2int[i]) + ',' + ''.join( str(x) for x in self.int2vec[i]) )
#         s += ';'.join(x) + ')'
#         return s

#     # representation string
#     def __repr__(self):
#         """
#         Returns the representation of the state space.

#         Returns:
#             str: The representation of the state space.
#         """
#         return self.make_str()
#     # print string
#     def __str__(self):
#         """
#         Returns the string representation of the state space.

#         Returns:
#             str: The string representation of the state space.
#         """
#         return self.make_str()



# def make_default_config2():
#     # 1. could run this script without writing to file if config.py DNE
#     # 2. check if config.py exists and avoid overwrite (y/n prompt, .bak, etc.)
#     # MJL 230802: we should be able to generate this automatically from
#     #             settings_registry if we provide default values.
#     s = """
# #==============================================================================#
# # Default phyddle config file                                                  #
# #==============================================================================#

# # external import
# import scipy.stats
# import scipy as sp

# # helper variables
# num_char = 3
# num_states = 2

# args = {

#     #-------------------------------#
#     # Project organization          #
#     #-------------------------------#
#     'proj'    : 'my_project',               # project name(s)
#     'step'    : 'SFTEP',                    # step(s) to run
#     'verbose' : True,                       # print verbose phyddle output?
#     'sim_dir' : '../workspace/simulate',    # directory for simulated data
#     'fmt_dir' : '../workspace/format',      # directory for tensor-formatted data
#     'trn_dir' : '../workspace/train',       # directory for trained network
#     'plt_dir' : '../workspace/plot',        # directory for plotted figures
#     'est_dir' : '../workspace/estimate',    # directory for predictions on new data
#     'log_dir' : '../workspace/log',         # directory for analysis logs

#     #-------------------------------#
#     # Multiprocessing               #
#     #-------------------------------#
#     'use_parallel'   : True,                # use multiprocessing to speed up jobs?
#     'num_proc'       : -2,                  # how many CPUs to use (-2 means all but 2)

#     #-------------------------------#
#     # Model Configuration           #
#     #-------------------------------#
#     'num_char'           : num_char,        # number of evolutionary characters
#     'num_states'         : num_states,      # number of states per character

#     #-------------------------------#
#     # Simulate Step settings        #
#     #-------------------------------#
#     'sim_command'       : 'python3 sim/MASTER/sim_one.py', # exact command, arg will be output file prefix
#     'sim_logging'       : 'verbose',        # verbose, compressed, or clean
#     'start_idx'         : 0,                # first simulation replicate index
#     'end_idx'           : 1000,             # last simulation replicate index

#     #-------------------------------#
#     # Format Step settings          #
#     #-------------------------------#
#     'tree_type'         : 'extant',         # use model with serial or extant tree
#     'chardata_format'   : 'nexus',
#     'tree_width_cats'   : [ 200, 500 ],     # tree width categories for phylo-state tensors
#     'tree_encode_type'  : 'height_brlen',   # how to encode phylo brlen? height_only or height_brlen
#     'char_encode_type'  : 'integer',        # how to encode discrete states? one_hot or integer
#     'param_est'        : [                  # model parameters to estimate (labels)
#         'w_0', 'e_0', 'd_0_1', 'b_0_1'
#     ],
#     'param_data'        : [],               # model parameters that are known (aux. data)
#     'tensor_format'     : 'hdf5',           # save as compressed HDF5 or raw csv
#     'save_phyenc_csv'   : False,            # save intermediate phylo-state vectors to file

#     #-------------------------------#
#     # Train Step settings           #
#     #-------------------------------#
#     'trn_objective'     : 'param_est',      # what is the learning task? param_est or model_test
#     'tree_width'        : 500,              # tree width category used to train network
#     'num_epochs'        : 20,               # number of training intervals (epochs)
#     'prop_test'         : 0.05,             # proportion of sims in test dataset
#     'prop_validation'   : 0.05,             # proportion of sims in validation dataset
#     'prop_calibration'  : 0.20,             # proportion of sims in CPI calibration dataset
#     'cpi_coverage'      : 0.95,             # coverage level for CPIs
#     'cpi_asymmetric'    : True,             # upper/lower (True) or symmetric (False) CPI adjustments
#     'batch_size'        : 128,              # number of samples in each training batch
#     'loss'              : 'mse',            # loss function for learning
#     'optimizer'         : 'adam',           # optimizer for network weight/bias parameters
#     'metrics'           : ['mae', 'acc'],   # recorded training metrics

#     #-------------------------------#
#     # Estimate Step settings        #
#     #-------------------------------#
#     'est_prefix'     : 'new.1',             # prefix for new dataset to predict

#     #-------------------------------#
#     # Plot Step settings            #
#     #-------------------------------#
#     'plot_train_color'      : 'blue',       # plot color for training data
#     'plot_test_color'       : 'purple',     # plot color for test data
#     'plot_val_color'        : 'red',        # plot color for validation data
#     'plot_aux_color'        : 'green',      # plot color for input auxiliary data
#     'plot_label_color'      : 'orange',     # plot color for labels (params)
#     'plot_est_color'        : 'black'       # plot color for estimated data/values

# }
# """
#     f = open('config_default.py', 'w')
#     f.write(s)
#     f.close()
#     return

# def sort_binary_vectors(binary_vectors):
#     """
#     Sorts a list of binary vectors.

#     The binary vectors are sorted first based on the number of "on" bits, and
#     then from left to right in terms of which bits are "on".

#     Args:
#         binary_vectors (List[List[int]]): The list of binary vectors to be sorted.

#     Returns:
#         List[List[int]]: The sorted list of binary vectors.
#     """
#     def count_ones(binary_vector):
#         """
#         Counts the number of "on" bits in a binary vector.

#         Args:
#             binary_vector (List[int]): The binary vector.

#         Returns:
#             int: The count of "on" bits.
#         """
#         return sum(binary_vector)

#     sorted_vectors = sorted(binary_vectors, key=count_ones)

#     for i in range(len(sorted_vectors)):
#         for j in range(i+1, len(sorted_vectors)):
#             if count_ones(sorted_vectors[j]) == count_ones(sorted_vectors[i]):
#                 for k in range(len(sorted_vectors[i])):
#                     if sorted_vectors[i][k] != sorted_vectors[j][k]:
#                         if sorted_vectors[j][k] > sorted_vectors[i][k]:
#                             sorted_vectors[i], sorted_vectors[j] = sorted_vectors[j], sorted_vectors[i]
#                         break

#     return sorted_vectors

# def powerset(iterable):
#     """
#     Generates all possible subsets (powerset) of the given iterable.

#     Args:
#         iterable: An iterable object.

#     Returns:
#         generator: A generator that yields each subset.
#     """
#     s = list(iterable)  # Convert the iterable to a list
#     return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


# def events2df(events):
#     """
#     Convert a list of Event objects to a pandas DataFrame.

#     This function takes a list of Event objects and converts it into a pandas
#     DataFrame. Each Event object represents a row in the resulting DataFrame,
#     with the Event attributes mapped to columns.

#     Args:
#         events (list): A list of Event objects.

#     Returns:
#         pandas.DataFrame: The resulting DataFrame with columns 'name', 'group',
#         'i', 'j', 'k', 'reaction', and 'rate'.
#     """
#     df = pd.DataFrame({
#         'name'     : [ e.name for e in events ],
#         'group'    : [ e.group for e in events ], 
#         'i'        : [ e.i for e in events ],
#         'j'        : [ e.j for e in events ],
#         'k'        : [ e.k for e in events ],
#         'reaction' : [ e.reaction for e in events ],
#         'rate'     : [ e.rate for e in events ]
#     })
#     return df

# def states2df(states):
#     """
#     Convert a States object to a pandas DataFrame.

#     This function takes a States object and converts it into a pandas DataFrame. The States object contains information about the state space, and the resulting DataFrame has columns 'lbl', 'int', 'set', and 'vec', representing the labels, integer representations, set representations, and vector representations of the states, respectively.

#     Args:
#         states (States): The States object to convert to a DataFrame.

#     Returns:
#         pandas.DataFrame: The resulting DataFrame with columns 'lbl', 'int', 'set', and 'vec'.
#     """
#     df = pd.DataFrame({
#         'lbl' : states.int2lbl,
#         'int' : states.int2int,
#         'set' : states.int2set,
#         'vec' : states.int2vec
#     })
#     return df



# def find_tree_width(num_taxa, max_taxa):
#     """Finds the CPSV width.

#     Returns the smallest suitable compact phylogenetic-state vector
#     representation such that num_taxa <= val for val in max_taxa.
#     Returns 0 if num_taxa == 0.
#     Returns -1 if num_taxa > max_taxa[-1].

#     Args:
#         num_taxa (int): the number of taxa in the raw dataset
#         max_taxa (list[int]):  a list of tree widths for CPSV encoding
    
#     Returns:
#         int: The smallest suitable tree width encoding
#     """
#     if num_taxa == 0:
#         return 0
#     elif num_taxa > max_taxa[-1]:
#         return -1
#     for i in max_taxa:
#         if num_taxa <= i:
#             return i
#     # should never call this
#     raise Exception('error in find_tree_width()', num_taxa, max_taxa)
#     #return -2


# def settings_to_str(settings, taxon_category):
#     """Convert settings dictionary and taxon category to a string representation.

#     This function takes a settings dictionary and a taxon category and converts
#     them into a comma-separated string representation. The resulting string
#     includes the keys and values of the settings dictionary, as well as the
#     taxon category.

#     Args:
#         settings (dict): The settings dictionary.
#         taxon_category (str): The taxon category.

#     Returns:
#         str: The string representation of the settings and taxon category.
#     """
#     s = 'setting,value\n'
#     s += 'model_name,' + settings['model_name'] + '\n'
#     s += 'model_type,' + settings['model_type'] + '\n'
#     s += 'replicate_index,' + str(settings['replicate_index']) + '\n'
#     s += 'taxon_category,' + str(taxon_category) + '\n'
#     return s

# def param_dict_to_str(params):
#     """Convert parameter dictionary to two string representations.

#     This function takes a parameter dictionary and converts it into two string
#     representations. The resulting strings includes the parameter names,
#     indices, and values. The first representation is column-based, the second
#     representation is row-based.

#     Args:
#         params (dict): The parameter dictionary.

#     Returns:
#         tuple: A tuple of two strings. The first string represents the parameter
#             values with indices, and the second string represents the parameter
#             names.
#     """
#     s1 = 'param,i,j,value\n'
#     s2 = ''
#     s3 = ''
#     for k,v in params.items():
#         for i,x in enumerate(v):
#             if len(v.shape) == 1:
#                 rate = np.round(x, NUM_DIGITS)
#                 s1 += '{k},{i},{i},{v}\n'.format(k=k,i=i,v=rate)
#                 s2 += '{k}_{i},'.format(k=k,i=i)
#                 s3 += str(rate) + ','
#             else:
#                 for j,y in enumerate(x):
#                     rate = np.round(y, NUM_DIGITS)
#                     s1 += '{k},{i},{j},{v}\n'.format(k=k,i=i,j=j,v=rate)
#                     s2 += '{k}_{i}_{j},'.format(k=k,i=i,j=j)
#                     s3 += str(rate) + ','

#     s4 = s2.rstrip(',') + '\n' + s3.rstrip(',') + '\n'
#     return s1,s4

# def clean_scientific_notation(s):
#     """Clean up a string representation of a number in scientific notation.

#     This function takes a string `s` representing a number in scientific
#     notation and removes unnecessary characters that indicate zero values.
#     The resulting string represents the number without trailing zeros in the
#     exponent.

#     Args:
#         s (str): The string representation of a number in scientific notation.

#     Returns:
#         str: The cleaned up string representation of the number.
#     """
#     return re.sub( '\.0+E\+0+', '', s)