#!/usr/bin/env python
"""
utilities
=========
Defines miscellaneous helper functions phyddle uses for pipeline steps.
Functions include argument parsing and checking, file conversion, managing
log files, and printing to screen.

Authors:   Michael Landis and Ammon Thompson
Copyright: (c) 2022-2025, Michael Landis and Ammon Thompson
License:   MIT
"""

# standard packages
import argparse
import os
import pkg_resources
import shutil
import tarfile
import platform
import re
import sys
import time
import dateutil
from datetime import datetime

# external packages
import dendropy as dp
import numpy as np
import pandas as pd

# phyddle imports
import __main__ as main
from . import PHYDDLE_VERSION, CONFIG_DEFAULT_FN

# Precision settings
OUTPUT_PRECISION = 16
PANDAS_FLOAT_FMT_STR = f'%.{OUTPUT_PRECISION}e'
NUMPY_FLOAT_FMT_STR = '{{:0.{:d}e}}'.format(OUTPUT_PRECISION)
np.set_printoptions(floatmode='maxprec', precision=OUTPUT_PRECISION)
pd.set_option('display.precision', OUTPUT_PRECISION)
pd.set_option('display.float_format', lambda x: f'{x:,.3f}')

# Tensorflow info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# run mode
INTERACTIVE_SESSION = not hasattr(main, '__file__')


##################################################

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
    for k, v in settings.items():
        # does this setting apply to the step?
        if step in v['step']:
            # get the setting from args to return
            ret[k] = args[k]

    # project directories
    for p in ['sim', 'fmt', 'trn', 'est', 'plt']:
        k = f'{p}_dir'
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
        'cfg':               {'step': '',       'type': str,  'section': 'Basic',  'default': 'config.py',     'help': 'Config file name', 'opt': 'c'},
        'step':              {'step': 'SFTEP',  'type': str,  'section': 'Basic',  'default': 'SFTEP',         'help': 'Pipeline step(s) defined with (S)imulate, (F)ormat, (T)rain, (E)stimate, (P)lot, or (A)ll', 'opt': 's'},
        'verbose':           {'step': 'SFTEP',  'type': str,  'section': 'Basic',  'default': 'T',             'help': 'Verbose output to screen?', 'bool': True, 'opt': 'v'},
        'make_cfg':          {'step': '',       'type': str,  'section': 'Basic',  'default': '__no_value__',  'help': 'Write default config file', 'const': CONFIG_DEFAULT_FN},
        'save_proj':         {'step': '',       'type': str,  'section': 'Basic',  'default': '__no_value__',  'help': 'Save and zip a project for sharing', 'const': 'project.tar.gz'},
        'load_proj':         {'step': '',       'type': str,  'section': 'Basic',  'default': '__no_value__',  'help': 'Unzip a shared project', 'const': 'project.tar.gz'},
        'clean_proj':        {'step': '',       'type': str,  'section': 'Basic',  'default': '__no_value__',  'help': 'Remove step directories for a project', 'const': '.'},
        'save_num_sim':      {'step': '',       'type': int,  'section': 'Basic',  'default': 10,              'help': 'Number of simulated examples to save with --save_proj'},
        'save_train_fmt':    {'step': '',       'type': str,  'section': 'Basic',  'default': 'F',             'help': 'Save formatted training examples with --save_proj? (not recommended)', 'bool': False},
        'output_precision':  {'step': 'SFTEP',  'type': int,  'section': 'Basic',  'default': 16,              'help': 'Number of digits (precision) for numbers in output files'},

        # directories
        'dir':         {'step': 'SFTEP',  'type': str,  'section': 'Workspace',  'default': './',   'help': 'Parent directory for all step directories unless step directory given'},
        'sim_dir':     {'step': 'SF',     'type': str,  'section': 'Workspace',  'default': None,   'help': 'Directory for raw simulated data'},
        'emp_dir':     {'step': 'SF',     'type': str,  'section': 'Workspace',  'default': None,   'help': 'Directory for raw empirical data'},
        'fmt_dir':     {'step': 'FTEP',   'type': str,  'section': 'Workspace',  'default': None,   'help': 'Directory for tensor-formatted data'},
        'trn_dir':     {'step': 'FTEP',   'type': str,  'section': 'Workspace',  'default': None,   'help': 'Directory for trained networks and training output'},
        'est_dir':     {'step': 'TEP',    'type': str,  'section': 'Workspace',  'default': None,   'help': 'Directory for new datasets and estimates'},
        'plt_dir':     {'step': 'P',      'type': str,  'section': 'Workspace',  'default': None,   'help': 'Directory for plotted results'},
        'log_dir':     {'step': 'SFTEP',  'type': str,  'section': 'Workspace',  'default': None,   'help': 'Directory for logs of analysis metadata'},
        'prefix':      {'step': 'SFTEP',  'type': str,  'section': 'Workspace',  'default': 'out',  'help': 'Prefix for all output unless step prefix given'},
        'sim_prefix':  {'step': 'SF',     'type': str,  'section': 'Workspace',  'default': None,   'help': 'Prefix for raw simulated data'},
        'emp_prefix':  {'step': 'SF',     'type': str,  'section': 'Workspace',  'default': None,   'help': 'Prefix for raw empirical data'},
        'fmt_prefix':  {'step': 'FTEP',   'type': str,  'section': 'Workspace',  'default': None,   'help': 'Prefix for tensor-formatted data'},
        'trn_prefix':  {'step': 'FTEP',   'type': str,  'section': 'Workspace',  'default': None,   'help': 'Prefix for trained networks and training output'},
        'est_prefix':  {'step': 'TEP',    'type': str,  'section': 'Workspace',  'default': None,   'help': 'Prefix for estimate results'},
        'plt_prefix':  {'step': 'P',      'type': str,  'section': 'Workspace',  'default': None,   'help': 'Prefix for plotted results'},

        # analysis options
        'use_parallel':  {'step': 'SF',   'type': str,   'section': 'Analysis',  'default': 'T',    'help': 'Use parallelization? (recommended)', 'bool': True},
        'use_cuda':      {'step': 'TE',   'type': str,   'section': 'Analysis',  'default': 'T',    'help': 'Use CUDA parallelization? (recommended; requires Nvidia GPU)', 'bool': True},
        'num_proc':      {'step': 'SFT',  'type': int,   'section': 'Analysis',  'default': -2,     'help': 'Number of cores for multiprocessing (-N for all but N)'},
        'no_emp':        {'step': '',     'type': None,  'section': 'Analysis',  'default': False,  'help': 'Disable Format/Estimate steps for empirical data?'},
        'no_sim':        {'step': '',     'type': None,  'section': 'Analysis',  'default': False,  'help': 'Disable Format/Estimate steps for simulated data?'},

        # simulation options
        'sim_command':     {'step': 'S',   'type': str,  'section': 'Simulate',  'default': None,      'help': 'Simulation command to run single job (see documentation)'},
        'sim_logging':     {'step': 'S',   'type': str,  'section': 'Simulate',  'default': 'clean',   'help': 'Simulation logging style', 'choices': ['clean', 'compress', 'verbose']},
        'start_idx':       {'step': 'SF',  'type': int,  'section': 'Simulate',  'default': 0,         'help': 'Start replicate index for simulated training dataset'},
        'end_idx':         {'step': 'SF',  'type': int,  'section': 'Simulate',  'default': 1000,      'help': 'End replicate index for simulated training dataset'},
        'sim_more':        {'step': 'S',   'type': int,  'section': 'Simulate',  'default': 0,         'help': 'Add more simulations with auto-generated indices'},
        'sim_batch_size':  {'step': 'S',   'type': int,  'section': 'Simulate',  'default': 1,         'help': 'Number of replicates per simulation command'},

        # formatting options
        'encode_all_sim':      {'step': 'F',     'type': str,   'section': 'Format',  'default': 'T',             'help': 'Encode all simulated replicates into tensor?', 'bool': True},
        'num_char':            {'step': 'SFTE',  'type': int,   'section': 'Format',  'default': None,            'help': 'Number of characters'},
        'num_states':          {'step': 'FTE',   'type': int,   'section': 'Format',  'default': None,            'help': 'Number of states per character'},
        'num_trees':           {'step': 'SFTE',  'type': int,   'section': 'Format',  'default': 1,               'help': 'Number of trees per dataset'},
        'min_num_taxa':        {'step': 'F',     'type': int,   'section': 'Format',  'default': 10,              'help': 'Minimum number of taxa allowed when formatting'},
        'max_num_taxa':        {'step': 'F',     'type': int,   'section': 'Format',  'default': 1000,            'help': 'Maximum number of taxa allowed when formatting'},
        'downsample_taxa':     {'step': 'FTE',   'type': str,   'section': 'Format',  'default': 'uniform',       'help': 'Downsampling strategy taxon count', 'choices': ['uniform']},
        'rel_extant_age_tol':  {'step': 'FTE',   'type': float, 'section': 'Format',  'default': 1E-5,            'help': 'Relative tolerance to determine if terminal taxa are extant (rel. age < tol).'},
        'tree_width':          {'step': 'FTEP',  'type': int,   'section': 'Format',  'default': 500,             'help': 'Width of phylo-state tensor'},
        'tree_encode':         {'step': 'FTEP',  'type': str,   'section': 'Format',  'default': 'extant',        'help': 'Encoding strategy for tree', 'choices': ['extant', 'serial']},
        'brlen_encode':        {'step': 'FTEP',  'type': str,   'section': 'Format',  'default': 'height_brlen',  'help': 'Encoding strategy for branch lengths', 'choices': ['height_only', 'height_brlen']},
        'char_encode':         {'step': 'FTE',   'type': str,   'section': 'Format',  'default': 'one_hot',       'help': 'Encoding strategy for character data', 'choices': ['one_hot', 'integer', 'numeric']},
        'param_est':           {'step': 'FTE',   'type': dict,  'section': 'Format',  'default': dict(),          'help': 'Model parameters and variables to estimate'},
        'param_data':          {'step': 'FTE',   'type': dict,  'section': 'Format',  'default': dict(),          'help': 'Model parameters and variables treated as data'},
        'char_format':         {'step': 'FTE',   'type': str,   'section': 'Format',  'default': 'nexus',         'help': 'File format for character data', 'choices': ['csv', 'nexus']},
        'tensor_format':       {'step': 'FTEP',  'type': str,   'section': 'Format',  'default': 'hdf5',          'help': 'File format for training example tensors', 'choices': ['csv', 'hdf5']},
        'save_phyenc_csv':     {'step': 'F',     'type': str,   'section': 'Format',  'default': 'F',             'help': 'Save encoded phylogenetic tensor encoding to csv?', 'bool': True},

        # training options
        'num_epochs':           {'step': 'TEP',    'type': int,    'section': 'Train',  'default': 50,             'help': 'Number of training epochs'},
        'num_early_stop':       {'step': 'TEP',    'type': int,    'section': 'Train',  'default': 3,              'help': 'Number of consecutive validation loss gains before early stopping'},
        'trn_batch_size':       {'step': 'TEP',    'type': int,    'section': 'Train',  'default': 512,            'help': 'Training batch sizes'},
        'prop_test':            {'step': 'FT',     'type': float,  'section': 'Train',  'default': 0.05,           'help': 'Proportion of data used as test examples (assess trained network performance)'},
        'prop_val':             {'step': 'T',      'type': float,  'section': 'Train',  'default': 0.05,           'help': 'Proportion of data used as validation examples (diagnose network overtraining)'},
        'prop_cal':             {'step': 'T',      'type': float,  'section': 'Train',  'default': 0.20,           'help': 'Proportion of data used as calibration examples (calibrate CPIs)'},
        'cpi_coverage':         {'step': 'T',      'type': float,  'section': 'Train',  'default': 0.95,           'help': 'Expected coverage percent for calibrated prediction intervals (CPIs)'},
        'cpi_asymmetric':       {'step': 'T',      'type': str,    'section': 'Train',  'default': 'T',            'help': 'Use asymmetric (True) or symmetric (False) adjustments for CPIs?', 'bool': True},
        'loss_numerical':       {'step': 'T',      'type': str,    'section': 'Train',  'default': 'mse',          'help': 'Loss function for real value estimates', 'choices': ['mse', 'mae']},
        'optimizer':            {'step': 'T',      'type': str,    'section': 'Train',  'default': 'adam',         'help': 'Method used for optimizing neural network', 'choices': ['adam', 'adadelta', 'adagrad', 'adamw', 'rmsprop', 'sgd']},
        'learning_rate':        {'step': 'T',      'type': float,  'section': 'Train',  'default': 0.001,          'help': 'Learning rate for optimizer'},
        'activation_func':      {'step': 'T',      'type': str,    'section': 'Train',  'default': 'relu',         'help': 'Activation function for all internal layers', 'choices': ['relu', 'leaky_relu', 'elu', 'tanh', 'sigmoid']}, 
        'log_offset':           {'step': 'FTEP',   'type': float,  'section': 'Train',  'default': 1.0,            'help': 'Offset size c when taking ln(x+c) for zero-valued variables'},
        'phy_channel_plain':    {'step': 'T',      'type': list,   'section': 'Train',  'default': [64, 96, 128],  'help': 'Output channel sizes for plain convolutional layers for phylogenetic state input'},
        'phy_channel_stride':   {'step': 'T',      'type': list,   'section': 'Train',  'default': [64, 96],       'help': 'Output channel sizes for stride convolutional layers for phylogenetic state input'},
        'phy_channel_dilate':   {'step': 'T',      'type': list,   'section': 'Train',  'default': [32, 64],       'help': 'Output channel sizes for dilate convolutional layers for phylogenetic state input'},
        'aux_channel':          {'step': 'T',      'type': list,   'section': 'Train',  'default': [128, 64, 32],  'help': 'Output channel sizes for dense layers for auxiliary data input'},
        'lbl_channel':          {'step': 'T',      'type': list,   'section': 'Train',  'default': [128, 64, 32],  'help': 'Output channel sizes for dense layers for label outputs'},
        'phy_kernel_plain':     {'step': 'T',      'type': list,   'section': 'Train',  'default': [3, 5, 7],      'help': 'Kernel sizes for plain convolutional layers for phylogenetic state input'},
        'phy_kernel_stride':    {'step': 'T',      'type': list,   'section': 'Train',  'default': [7, 9],         'help': 'Kernel sizes for stride convolutional layers for phylogenetic state input'},
        'phy_kernel_dilate':    {'step': 'T',      'type': list,   'section': 'Train',  'default': [3, 5],         'help': 'Kernel sizes for dilate convolutional layers for phylogenetic state input'},
        'phy_stride_stride':    {'step': 'T',      'type': list,   'section': 'Train',  'default': [3, 6],         'help': 'Stride sizes for stride convolutional layers for phylogenetic state input'},
        'phy_dilate_dilate':    {'step': 'T',      'type': list,   'section': 'Train',  'default': [3, 5],         'help': 'Dilation sizes for dilate convolutional layers for phylogenetic state input'},

        # estimating options
        'warn_aux_outlier':     {'step': 'FEP',    'type': float,  'section': 'Estimate',  'default': 0.0001,      'help': 'Percentile to detect extreme empirical auxiliary (abs.) values.'},
        'warn_lbl_outlier':     {'step': 'FEP',    'type': float,  'section': 'Estimate',  'default': 0.01,        'help': 'Percentile to detect extreme empirical label (abs.) values.'},

        # plotting options
        'plot_train_color'  : {'step': 'P', 'type': str,    'section': 'Plot', 'default': 'blue',       'help': 'Plotting color for training data elements'},
        'plot_test_color'   : {'step': 'P', 'type': str,    'section': 'Plot', 'default': 'purple',     'help': 'Plotting color for test data elements'},
        'plot_val_color'    : {'step': 'P', 'type': str,    'section': 'Plot', 'default': 'red',        'help': 'Plotting color for validation data elements'},
        'plot_label_color'  : {'step': 'P', 'type': str,    'section': 'Plot', 'default': 'orange',     'help': 'Plotting color for label elements'},
        'plot_phy_color'    : {'step': 'P', 'type': str,    'section': 'Plot', 'default': 'teal',       'help': 'Plotting color for phylogenetic data elements'},
        'plot_aux_color'    : {'step': 'P', 'type': str,    'section': 'Plot', 'default': 'green',      'help': 'Plotting color for auxiliary data elements'},
        'plot_emp_color'    : {'step': 'P', 'type': str,    'section': 'Plot', 'default': 'black',      'help': 'Plotting color for empirical elements'},
        'plot_num_scatter'  : {'step': 'P', 'type': int,    'section': 'Plot', 'default': 50,           'help': 'Number of examples in scatter plot'},
        'plot_min_emp'      : {'step': 'P', 'type': int,    'section': 'Plot', 'default': 10,           'help': 'Minimum number of empirical datasets to plot densities'},
        'plot_num_emp'      : {'step': 'P', 'type': int,    'section': 'Plot', 'default': 5,            'help': 'Number of empirical results to plot'},
        'plot_pca_noise'    : {'step': 'P', 'type': float,  'section': 'Plot', 'default': 0.0,          'help': 'Scale of Gaussian noise to add to PCA plot'},
    }

    # Developer note: uncomment to export settings to file
    # export_settings_to_sphinx_table(settings)

    return settings


class CustomHelpFormatter(argparse.HelpFormatter):
    """Replaces default argparse help formatter
    This is modified ChatGPT code designed to suppress arguments with the const
    field set (e.g. make_cfg, load_proj, etc.) from displaying useless
    option/flag arguments when calling phyddle --help."""
    def _format_action_invocation(self, action):
        if not action.option_strings:
            return super(CustomHelpFormatter, self)._format_action_invocation(action)

        # Custom handling for actions with `const` values
        if action.const is not None:
            # This is where make_cfg, load_proj, etc. help tags are cleaned
            return ', '.join(action.option_strings)
        
        default = self._get_default_metavar_for_optional(action)
        args_string = self._format_args(action, default)
        return ', '.join(action.option_strings) + ' ' + args_string

def export_settings_to_sphinx_table(settings, csv_fn='phyddle_settings.csv'):
    """Writes all phyddle settings to file as Sphinx-formatted table """

    # setting header
    s = 'Setting|Step(s)|Type|Description\n'
    for k, v in settings.items():

        # setting name
        s_name = f'``{k}``'

        # setting step
        s_step = 'SFTEP'
        if v['step'] is None:
            s_step = '––'
        for i, this_step in enumerate(s_step):
            if this_step not in v['step']:
                s_step = s_step.replace(this_step, '–')

        # setting type
        if v['type'] is None:
            s_type = '––'
        elif v['type'].__name__ == 'list':
            s_elt_type = type(v['default'][0]).__name__
            s_type = f'*{s_elt_type}[]*'
        else:
            s_type = f'*{v["type"].__name__}*'

        # setting desc
        s_desc = v['help']
        if s_name == 'proj':
            s_desc += ', *see detailed description* [:ref:`link <setting_description_proj>`]'
        elif s_name == 'step':
            s_desc += ', *see detailed description* [:ref:`link <setting_description_step>`]'

        # setting row
        s += f'{s_name}|{s_step}|{s_type}|{s_desc}\n'

    f = open(csv_fn, 'w')
    f.write(s)
    f.close()
    return


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
    desc = ('Phylogenetic model exploration with deep learning.'
            'models. Visit https://phyddle.org for documentation.')
    parser = argparse.ArgumentParser(description=desc,
                                     formatter_class=CustomHelpFormatter)

    # read settings registry and populate argument parser
    settings = settings_registry()
    for k, v in settings.items():
        arg_opt = []
        if 'opt' in v:
            arg_opt.append(f'-{v["opt"]}')
        arg_dest = k
        arg_opt.append(f'--{k}')
        arg_help = v['help']
        arg_type = v['type']
        arg_const = None
        if 'const' in v:
            arg_const = v['const']

        if arg_type is None:
            parser.add_argument(*arg_opt, action='store_true', help=arg_help)
        elif arg_const is not None:
            # used for special flags, --save_proj, --load_proj, --clean_proj
            parser.add_argument(*arg_opt, nargs='?', const=arg_const,
                                default=v['default'], type=arg_type, metavar='',
                                help=arg_help)
        else:
            if 'choices' in v:
                arg_choices = v['choices']
                parser.add_argument(*arg_opt, dest=arg_dest,
                                    type=arg_type, choices=arg_choices,
                                    help=arg_help, metavar=None)
            else:
                parser.add_argument(*arg_opt, dest=arg_dest,
                                    type=arg_type, help=arg_help, metavar='')

    # COMMAND LINE SETTINGS
    args = parser.parse_args(args)

    # CHECK IF IN RUN MODE OR TOOL MODE
    run_mode = True
    if args.make_cfg != '__no_value__' or \
            args.save_proj != '__no_value__' or \
            args.load_proj != '__no_value__' or \
            args.clean_proj != '__no_value__':
        run_mode = False

    # DEFAULT CONFIG FILE SETTINGS
    # make/overwrite default config, if requested
    if args.make_cfg != '__no_value__':
        make_config_fn = args.make_cfg
        if arg_overwrite and args.cfg is not None:
            make_config_fn = args.cfg
        make_default_config(make_config_fn)
        print_str(f"Created default config as '{make_config_fn}' ...")
        sys.exit()

    # if not os.path.exists(CONFIG_DEFAULT_FN):
    # msg = f"Default config file '{CONFIG_DEFAULT_FN} not found. Creating "
    # msg += "default config file in current directory."
    # print_warn(msg)
    # make_default_config(CONFIG_DEFAULT_FN)

    # load config into namespace
    # TODO: can this be deleted?
    default_args = {}
    # if os.path.exists(CONFIG_DEFAULT_FN):
    #     namespace = {}
    #     with open(CONFIG_DEFAULT_FN) as file:
    #         code = file.read()
    #         exec(code, namespace)
    #     # move imported args into local variable
    #     default_args = namespace['args']

    # PROJECT CONFIG FILE SETTINGS
    if arg_overwrite and args.cfg is not None:
        config_fn = args.cfg

    namespace = {'args': {}}

    if run_mode and not os.path.exists(config_fn):
        msg = (f"Project config file '{config_fn}' not found. If you have a "
               "project config file, please verify access to the file. There "
               "are two easy ways to create a user config file: (1) copy the "
               f"default config file '{CONFIG_DEFAULT_FN}' and modify it as "
               "needed, or (2) download an existing project config file from "
               "https://github.com/mlandis/phyddle/tree/main/scripts.")
        print_err(msg)
        sys.exit()
    elif os.path.exists(config_fn):
        # namespace = {}
        with open(config_fn) as file:
            code = file.read()
            exec(code, namespace)

    # move imported args into local variable
    file_args = namespace['args']

    # MERGE SETTINGS
    # merge default, user_file, and user_cmd settings
    m = reconcile_settings(settings, default_args, file_args, args)

    # fix convert string-valued bool to true bool
    m = fix_arg_bool(m)

    # update steps
    if m['step'] == 'A':
        m['step'] = 'SFTEP'

    # print header?
    verbose = m['verbose']
    if verbose:
        print(phyddle_header('title'))

    # check arguments are valid
    if run_mode:
        check_args(m)

    # handle project prefix
    m = set_step_args(m)

    # set steps & projects
    # m = add_step_proj(m)

    # add session info
    date_obj = datetime.now()
    m['date'] = date_obj.strftime("%y%m%d_%H%M%S")
    m['job_id'] = generate_random_hex_string(7)

    # update output precision
    global OUTPUT_PRECISION, PANDAS_FLOAT_FMT_STR, NUMPY_FLOAT_FMT_STR
    OUTPUT_PRECISION = m['output_precision']
    PANDAS_FLOAT_FMT_STR = f'%.{OUTPUT_PRECISION}e'
    NUMPY_FLOAT_FMT_STR = '{{:0.{:d}e}}'.format(OUTPUT_PRECISION)
    np.set_printoptions(floatmode='maxprec', precision=OUTPUT_PRECISION)
    pd.set_option('display.precision', OUTPUT_PRECISION)
    pd.set_option('display.float_format', lambda x: f'{x:,.6f}')

    # PROJECT MANAGEMENT
    # save project
    if args.save_proj != '__no_value__':
        save_project(m, args, config_fn)
        sys.exit()

    # load project
    if args.load_proj != '__no_value__':
        load_project(args)
        sys.exit()

    # clean project
    if args.clean_proj != '__no_value__':
        clean_project(m, args)
        sys.exit()

    if not run_mode:
        sys.exit()

    # return new args
    return m


def save_project(m, args, config_fn):
    """Save a project as a tarball for sharing."""
    print_str(f"Saving project as '{args.save_proj}'...")
    tarball = tarfile.open(args.save_proj, 'w:gz')
    print_str(f"  ▪ {config_fn}")
    tarball.add(config_fn)
    for tok in m['sim_command'].split(' '):
        if os.path.exists(tok):
            print_str(f"  ▪ {tok}")
            tarball.add(tok)
    if os.path.isdir(m['sim_dir']):
        sim_files = os.listdir(m['sim_dir'])
        sim_files = sorted_nicely(sim_files)
        eg_files = [ f for f in sim_files if f.endswith('.labels.csv') ]
        num_save_sim = m['save_num_sim']
        eg_files = eg_files[:num_save_sim]
        for f in eg_files:
            tok = f.split('.')
            idx_files = [ f for f in sim_files if f.startswith('.'.join(tok[0:2])+'.') ]
            for g in idx_files:
                tarball.add(f'{m["sim_dir"]}/{g}')
        if len(eg_files) > 0:
            print_str(f"  ▪ {m['sim_dir']}   [num. examples: {len(eg_files)}]")
    if os.path.isdir(m['fmt_dir']):
        for f in os.listdir(m['fmt_dir']):
            f_tok = f.split('.')
            has_empirical = (f_tok[1] == 'empirical')
            has_test = (f_tok[1] == 'test')
            has_train = (f_tok[1] == 'train') and m['save_train_fmt']
            valid_prefix = f_tok[0] == m['fmt_prefix']
            valid_ext = f_tok[-1] == 'csv' or f_tok[-1] == 'hdf5'
            valid_data = has_test or has_empirical or has_train
            if valid_prefix and valid_data and valid_ext:
                tarball.add(f'{m["fmt_dir"]}/{f}')
        if not m['save_train_fmt']:
            print_str(f"  ▪ {m['fmt_dir']}     [train not saved]")
        else:
            print_str(f"  ▪ {m['fmt_dir']}")
    if os.path.isdir(m['emp_dir']):
        tarball.add(m['emp_dir'])
        print_str(f"  ▪ {m['emp_dir']}")
    if os.path.isdir(m['trn_dir']):
        tarball.add(m['trn_dir'])
        print_str(f"  ▪ {m['trn_dir']}")
    if os.path.isdir(m['est_dir']):
        tarball.add(m['est_dir'])
        print_str(f"  ▪ {m['est_dir']}")
    if os.path.isdir(m['plt_dir']):
        tarball.add(m['plt_dir'])
        print_str(f"  ▪ {m['plt_dir']}")
    tarball.close()
    print_str("... done!")
    return


def load_project(args):
    """Load a project from a tarball."""
    print_str(f"Loading project '{args.load_proj}'...")
    tarball = tarfile.open(args.load_proj, 'r:gz')
    for f in tarball:
        try:
            tarball.extract(f)
        except IOError:
            os.remove(f.name)
            tarball.extract(f)
        finally:
            os.chmod(f.name, f.mode)
    tarball.close()
    print_str("... done!")
    return


def clean_project(m, args):
    """Clean a project."""
    print_str(f"Cleaning project in directory '{args.clean_proj}'...")
    for d in ['sim_dir', 'fmt_dir', 'trn_dir', 'est_dir', 'plt_dir']:
        if os.path.isdir(m[d]):
            shutil.rmtree(m[d])
    print_str("... done!")
    return


def fix_arg_bool(m):
    """Convert bool-str arguments to True/False bool."""
    settings = settings_registry()
    for k, v in settings.items():
        if 'bool' in v and type(m[k]) is not str:
            raise Exception(f"Invalid argument: {k} must be a string")
        elif 'bool' in v and type(m[k]) is str:
            arg_val = m[k]
            arg_val_new = str2bool(arg_val)
            if arg_val_new is None:
                raise ValueError(f'{arg_val} invalid value for {k}')
            m[k] = arg_val_new
    return m


def str2bool(x):
    """Convert a str value to True/False"""
    if x.lower() in ['true', 'yes', 't', 'y', '1']:
        return True
    elif x.lower() in ['false', 'no', 'f', 'n', '0']:
        return False
    else:
        return None


def set_step_args(args):
    """Sets step-specific arguments for a project."""

    proj_dict = {
        'sim': 'simulate',
        'emp': 'empirical',
        'fmt': 'format',
        'trn': 'train',
        'est': 'estimate',
        'plt': 'plot',
        'log': 'log'
    }

    for k, v in proj_dict.items():
        k_prefix = f'{k}_prefix'
        k_dir = f'{k}_dir'
        if k_prefix not in args:
            args[k_prefix] = args['prefix']
        if k_dir not in args:
            args[k_dir] = args['dir'] + '/' + v
            args[k_dir] = args[k_dir].replace('//', '/')

    return args


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
    if not all([s in 'ASFTEP' for s in args['step']]):
        print_err("step must contain only letters 'ASFTEP'", exit=True)
    if args['sim_logging'] not in ['clean', 'verbose', 'compress']:
        print_err("sim_logging must be 'clean', 'verbose', or 'compress'",
                  exit=True)
    if args['tree_encode'] not in ['serial', 'extant']:
        print_err("tree_encode must be 'serial' or 'extant'", exit=True)
    if args['brlen_encode'] not in ['height_only', 'height_brlen']:
        print_err("brlen_encode must be 'height_only' or 'height_brlen'",
                  exit=True)
    if args['char_encode'] not in ['one_hot', 'integer', 'numeric']:
        print_err("char_encode must be 'one_hot', 'integer', or 'numeric'",
                  exit=True)
    if args['tensor_format'] not in ['csv', 'hdf5']:
        print_err("tensor_format must be 'csv' or 'hdf5'", exit=True)
    if args['char_format'] not in ['csv', 'nexus']:
        print_err("char_format must be 'csv' or 'nexus'", exit=True)
    if args['downsample_taxa'] not in ['uniform']:
        print_err("downsample_taxa must be 'uniform'", exit=True)

    # numerical values
    if args['start_idx'] < 0:
        print_err("start_idx must be >= 0", exit=True)
    if args['end_idx'] < 0:
        print_err("end_idx must be >= 0", exit=True)
    if args['start_idx'] > args['end_idx']:
        print_err("start_idx must be <= end_idx", exit=True)
    if args['sim_more'] < 0:
        print_err("sim_more must be >= 0", exit=True)
    if args['min_num_taxa'] < 0:
        print_err("min_num_taxa must be >= 0", exit=True)
    if args['max_num_taxa'] < 0:
        print_err("max_num_taxa must be >= 0", exit=True)
    if args['min_num_taxa'] > args['max_num_taxa']:
        print_err("min_num_taxa must be <= max_num_taxa", exit=True)
    if args['tree_width'] < 0:
        print_err("tree_width must be >= 0", exit=True)
    if args['num_states'] < 0:
        print_err("num_states must be >= 0", exit=True)
    if args['num_char'] < 0:
        print_err("num_char must be >= 0", exit=True)
    if args['num_trees'] < 0 or args['num_trees'] > 1:
        print_err("num_trees must be 0 or 1", exit=True)
    if args['num_epochs'] < 0:
        print_err("num_epochs must be >= 0", exit=True)
    if args['num_early_stop'] < 0:
        print_err("num_early_stop must be >= 0", exit=True)
    if args['trn_batch_size'] <= 0:
        print_err("trn_batch_size must be > 0", exit=True)
    if args['sim_batch_size'] <= 0:
        print_err("sim_batch_size must be > 0", exit=True)
    if args['cpi_coverage'] < 0. or args['cpi_coverage'] > 1.:
        print_err("cpi_coverage must be between 0 and 1", exit=True)
    if args['warn_aux_outlier'] < 0. or args['warn_aux_outlier'] > 1.:
        print_err("warn_aux_outlier must be between 0 and 1", exit=True)
    if args['warn_lbl_outlier'] < 0. or args['warn_lbl_outlier'] > 1.:
        print_err("warn_lbl_outlier must be between 0 and 1", exit=True)
    if args['prop_test'] < 0. or args['prop_test'] > 1.:
        print_err("prop_test must be between 0 and 1", exit=True)
    if args['prop_val'] < 0. or args['prop_val'] > 1.:
        print_err("prop_val must be between 0 and 1", exit=True)
    if args['prop_cal'] < 0. or args['prop_cal'] > 1.:
        print_err("prop_cal must be between 0 and 1", exit=True)
    if args['plot_pca_noise'] < 0.:
        print_err("plot_pca_noise must be >= 0", exit=True)
        
    for k in args['param_est'].keys():
        if k in args['param_data']:
            print_err(f"Parameter '{k}' cannot be in both param_est and param_data", exit=True)

    for k in args['param_data'].keys():
        if k in args['param_est']:
            print_err(f"Parameter '{k}' cannot be in both param_est and param_data", exit=True)
            
    for k,v in args['param_est'].items():
        if v not in ['cat', 'num']:
            print_err(f"param_est[{k}] must be 'cat' or 'num'", exit=True)
            
    for k,v in args['param_data'].items():
        if v not in ['cat', 'num']:
            print_err(f"param_data[{k}] must be 'cat' or 'num'", exit=True)

    unused_args = []
    settings = settings_registry()
    for k, v in args.items():
        if k not in settings.keys():
            unused_args.append(k)

    if len(unused_args) > 0:
        print_warn(
            "These applied settings are unknown and will not be used: " + ' '.join(
                unused_args))

    # done
    return


# def add_step_proj(args): #steps, proj):
#     """Manages project directories for steps.
#
#     This function determines which step will be used from args. Next it
#     processes the project string to determine the project(s) name(s) across
#     all steps. Last, it creates step-specific project directories and stores
#     them back into args.
#
#     Args:
#         args (dict): Un-updated phyddle settings dictionary
#
#     Returns:
#         dict: Updated the phyddle settings dictionary.
#
#     """
#     # get relevant args
#     steps = args['step']
#     proj = args['proj']
#
#     # different ways of naming steps
#     d_map = { 'S': ('sim', 'simulate'),
#               'F': ('fmt', 'format'),
#               'T': ('trn', 'train'),
#               'E': ('est', 'estimate'),
#               'P': ('plt', 'plot'),
#               'L': ('log', 'log') }
#
#     # parse input string
#     d_toks = {}
#     proj_toks = proj.split(',')
#     for p in proj_toks:
#         if ':' not in p:
#             d_toks['A'] = p
#         else:
#             k,v = p.split(':')
#             d_toks[k] = v
#
#     # handle all-step ('A') first
#     d_arg = {}
#     if 'A' in d_toks.keys():
#         steps = 'SFTEPL'
#         for i in ['S', 'F', 'T', 'E', 'P', 'L']:
#             k = d_map[i][0]
#             d_arg[k] = d_toks['A']
#
#     # overwrite with named steps
#     k_change = [ k for k in d_toks.keys() if k in 'SFTEPL' ]
#     for k in k_change:
#         d_arg[ d_map[k][0] ] = d_toks[k]
#
#     # verify all steps are covered
#     for s in steps:
#         if d_map[s][0] not in d_arg.keys():
#             msg = f"Step {s} ({d_map[s][1]}) has no assigned project name"
#             raise ValueError(msg)
#
#     for k in d_arg.keys():
#         k_str = k + '_proj'
#         args[k_str] = d_arg[k]
#
#     return args

def make_default_config(config_fn):
    """Generate default config file.

    Processes all items in the settings_registry to create a formatted
    default config dictionary. Writes config to file.

    """

    # get settings registry
    settings = settings_registry()

    # order sections
    section_settings = dict()
    section_settings['Basic'] = {}
    section_settings['Analysis'] = {}
    section_settings['Workspace'] = {}
    section_settings['Simulate'] = {}
    section_settings['Format'] = {}
    section_settings['Train'] = {}
    section_settings['Estimate'] = {}
    section_settings['Plot'] = {}

    # populate settings by section
    for k, v in settings.items():
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
    len_key = 20                  # "'num_epochs'".ljust(len_key, ' ')
    len_value = 20                # str(20).ljust(len_value, ' ')
    # len_help = 32               # "Number of training examples"
    len_punct = len_indent + len_assign + len_comment
    len_line = 80

    section_str = {}
    for i, (k1, v1) in enumerate(section_settings.items()):
        # section header
        s_sect = "  #-------------------------------#\n"
        s_sect += "  # " + str(k1).ljust(30, ' ') + "#\n"
        s_sect += "  #-------------------------------#\n"

        # max widths for sections
        max_key, max_value, max_help = 0, 0, 0

        # info for sections
        v_key, v_value, v_help = [], [], []

        # get key,value,help and max sizes per section
        for k2, v2 in v1.items():
            # key
            s_key = f"'{str(k2)}'"
            v_key.append(s_key)
            max_key = max(max_key, len(s_key))

            # value
            s_value = str(v2['default'])
            if v2['type'] is str and s_value != 'None':
                s_value = "'" + s_value + "'"
            v_value.append(s_value + ',')
            max_value = max(max_value, len(s_value))

            # help
            s_help = v2['help']
            v_help.append(s_help)
            max_help = max(max_help, len(s_help))

        # make key,value,help strings for each section entry
        for ki, vi, hi in zip(v_key, v_value, v_help):
            width_key = max(len_key, max_key)
            width_value = max(len_value, max_value)
            width_help = len_line - (width_key + width_value + len_punct)

            s_tok = list()
            s_tok.append(s_indent)
            s_tok.append(ki.ljust(width_key, ' '))
            s_tok.append(s_assign)
            s_tok.append(vi.ljust(width_value, ' '))
            s_tok.append(s_comment)
            s_tok.append(hi.ljust(width_help, ' '))

            s_sect += ''.join(s_tok) + '\n'

        if len(v1) == 0:
            s_sect += "  # none currently\n"

        section_str[k1] = s_sect

    # build file content
    s_cfg = "#====================================================================#\n"
    s_cfg += "# Default phyddle config file                                        #\n"
    s_cfg += "#====================================================================#\n"
    s_cfg += "\n"
    s_cfg += "args = {\n"
    for k, v in section_str.items():
        s_cfg += v + '\n'
    s_cfg += "}\n"

    # write file content
    f = open(config_fn, 'w')
    f.write(s_cfg)
    f.close()

    # done
    return


# update arguments from defaults, when provided
def reconcile_settings(settings_args, default_args, file_args, cmd_args):
    """Reconciles settings from all sources.

        Settings are applied and overwritten in this order:
        1. default settings from phyddle source
        2. default settings from file (config_default.py)
        3. analysis settings from file (e.g. config_<example>.py)
        4. command line settings (e.g. --sim_more 200)

        Settings are applied to default settings before being returned.

        Args:
            settings_args (dict): Default settings from phyddle
            default_args (dict): Default settings from file
            file_args (dict): Config file settings
            cmd_args (Namespace): Command line settings
        
        Returns:
            default_args (dict): Updated default settings.

        """

    args = {}

    # (1) start with hard-coded phyddle default settings
    for k, v in settings_args.items():
        if 'default' in v:
            if v['default'] is not None:
                args[k] = v['default']

    # (2) overwrite with default file args
    for k, v in default_args.items():
        if v is not None:
            args[k] = v

    # (3) overwrite with specific file args
    for k, v in file_args.items():
        if v is not None:
            args[k] = v

    # (4) overwrite with command line args
    for k in settings_args.keys():
        if k in cmd_args:
            v = getattr(cmd_args, k)
            if v is not None:
                args[k] = v

    return args


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


##################################################

###################
# GENERAL HELPERS #
###################

def set_seed(seed_value):
    # Tensorflow (old): https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
    # PyTorch (current): https://pytorch.org/docs/stable/notes/randomness.html#reproducibility

    import torch
    import numpy
    import random
    import os

    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    numpy.random.seed(seed_value)
    torch.manual_seed(seed_value)

    return


def get_time():
    """Get current clock time."""
    t = time.localtime()
    s = time.strftime("%H:%M:%S", t)
    return time.mktime(t), s


def get_time_str():
    """Make time string."""
    return get_time()[1]


def get_time_diff(start_time, end_time):
    """Make time-difference string.
    
    Args:
        start_time (float): start time
        end_time (float): end time

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


def get_num_tree_col(tree_encode, brlen_encode):
    """Gets number of tree columns (tree width).
    
    Computes number of columns for encoding tree information using a compact
    phylogenetic vector + states (CPV+S) format.

    CBLV (2 columns) is used if tree_encode == 'serial'. CDV (1 column) is used
    if tree_encode == 'extant'.

    No extra columns are used if brlen_encode == 'height_only'. Two extra
    columns are used if brlen_encode == 'height_brlen'.

    Args:
        tree_encode (str): Use CBLV (serial) or CDV (extant) for tree encoding.
        brlen_encode (str): Use height-only or height + brlen encoding.

    Returns:
        int: The number of columns for the CPV encoding.

    """

    num_tree_col = 0

    if tree_encode == 'serial':
        num_tree_col = 2
    elif tree_encode == 'extant':
        num_tree_col = 1

    if brlen_encode == 'height_only':
        num_tree_col += 0
    elif brlen_encode == 'height_brlen':
        num_tree_col += 2

    return num_tree_col


def get_num_char_col(state_encode_type, num_char, num_states):
    """Gets number of character columns.
    
    Computes number of columns for encoding state information using a compact
    phylogenetic vector + states (CPV+S) format.

    Integer encoding uses 1 column per character, with any number of states
    per character.

    One-hot encoding uses k columns per character, where k is the number of
    states.

    Args:
        state_encode_type (str): Use integer or one_hot encoding
        num_char (int): The number of characters in the matrix.
        num_states (int): The number of states per character.
        
    Returns:
        int: The number of columns for the +S encoding.
    
    """

    # num_char_col = 0

    if state_encode_type == 'integer' or state_encode_type == 'numeric':
        num_char_col = num_char
    elif state_encode_type == 'one_hot':
        num_char_col = num_char * num_states
    else:
        raise ValueError(
            f'"{state_encode_type}" is not recognized char_encode type')

    return num_char_col


def append_row(df, row):
    """Appends row to Pandas DataFrame
    
    Args:
        df (pd.DataFrame): Dataframe to be updated with K columns
        row (list): Row of length K to append to df

    Returns
        pd.DataFrame: The original dataframe with row appended to the end
    """
    assert (len(row) == len(df.columns))
    df.loc[len(df)] = row
    return df


def sorted_nicely(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key=alphanum_key)

##################################################


################
# FILE HELPERS #
################

def read_csv_as_pandas(fn):
    """Reads a CSV file into a pandas DataFrame.

    Args:
        fn (str): The file name or path of the CSV file.

    Returns:
        pd.DataFrame: The parsed CSV file as a pandas DataFrame.

    """
    df = None
    if os.path.exists(fn):
        df = pd.read_csv(fn)
        if len(df.columns) == 0:
            df = None

    return df


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

        if phy is None:
            phy = phy_tmp

    return phy


##################################################

#####################
# FORMAT CONVERTERS #
#####################

def convert_csv_to_array(dat_fn, char_encode, num_states=None):
    """Converts CSV to array format."""
    if char_encode == 'numeric' or char_encode == 'integer':
        dat = convert_csv_to_numeric_array(dat_fn)   #, num_states)
    elif char_encode == 'one_hot':
        dat = convert_csv_to_onehot_array(dat_fn, num_states)
    else:
        return NotImplementedError
    return dat


def convert_csv_to_numeric_array(dat_fn):
    """Converts a csv file to an integer-encoded pandas DataFrame."""

    try:
        # dat is pandas.DataFrame if non-empty
        dat = pd.read_csv(dat_fn, delimiter=',', index_col=0, header=None).T
    except pd.errors.EmptyDataError:
        # dat is None if empty
        return None
    # return
    return dat


def convert_csv_to_onehot_array(dat_fn, num_states):
    """Converts a csv file to an integer-encoded pandas DataFrame."""

    try:
        # dat is pandas.DataFrame if non-empty
        dat_raw = pd.read_csv(dat_fn, delimiter=',', index_col=0, header=None).T
    except pd.errors.EmptyDataError:
        return None

    # get num taxa (columns)
    num_taxa = dat_raw.shape[1]

    # check/unify number of state per row
    if type(num_states) is int:
        num_states = [num_states] * dat_raw.shape[0]

    assert (dat_raw.shape[0] == len(num_states))
    assert (all([type(i) is int for i in num_states]))

    # make csv
    num_rows = sum(num_states)
    zero_data = np.zeros(shape=(num_rows, num_taxa), dtype='int')
    dat = pd.DataFrame(zero_data, columns=dat_raw.columns)

    # do one-hot encoding
    j = 0
    for i, ns in enumerate(num_states):
        k = j + num_states[i]
        dat.iloc[j:k, :] = to_categorical(dat_raw.iloc[i, :], num_classes=ns,
                                          dtype='int').T
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

    # check that file is valid
    if len(lines) == 0:
        return None
    if lines[0].upper() != '#NEXUS\n':
        return None

    # process file
    found_matrix = False
    num_taxa = 0
    num_char = 0
    taxon_idx = 0
    taxon_names = []
    dat = None
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
                name = tok[0]
                state = tok[1]
                taxon_names.append(name)
                dat[:, taxon_idx] = [int(z) for z in state]
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
        num_states (int): Number of states to one-hot encode.

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

    # check that file is valid
    if len(lines) == 0:
        return None
    if lines[0].upper() != '#NEXUS\n':
        return None

    # helper variables
    found_matrix = False
    num_taxa = 0
    # num_char = 0
    num_one_hot = 0
    taxon_idx = 0
    taxon_names = []
    dat = None

    # print('\n'.join(lines))
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
                v = [int(z) for z in state]
                for i, j in enumerate(v):
                    state_idx = i * num_states + j
                    dat[state_idx, taxon_idx] = 1
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
    nex_str = nex_file.readlines()[3]
    # MJL add regex string qualifier
    matches = re.findall(
        pattern=r'([0-9]+)\[\&type="([A-Z]+)",location="([0-9]+)"',
        string=nex_str)
    num_taxa = len(matches)
    nex_file.close()

    # generate taxon-state data
    # d = {}
    s_state_str = ''
    for i, v in enumerate(matches):
        taxon = v[0]
        state = int(v[2])
        vec_str = ''.join([str(x) for x in int2vec[state]])
        # d[ taxon ]   = vec_str
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


def encode_cpvs(phy, dat, tree_width, tree_type,
                tree_encode_type, idx, rescale=True):
    """
    Encode Compact Phylogenetic Vector + States (CPV+S) array

    This function encodes the dataset into Compact Bijective Ladderized
    Vector + States (CBLV+S) when tree_type is 'serial' or Compact
    Diversity-Reordered Vector + States (CDV+S) when tree_type is 'extant'.

    Arguments:
        phy (dendropy.Tree):     phylogenetic tree
        dat (numpy.array):       character data
        tree_width (int):        number of columns (max. num. taxa)
                                 in CPVS array
        tree_type (str):         type of the tree ('serial' or 'extant')
        tree_encode_type (str):  type of tree encoding ('height_only' or
                                 'height_brlen')
        idx (int):               replicate index
        rescale (bool):          set tree height to 1 then encode, if True

    Returns:
        cpvs (numpy.array):      CPV+S encoded tensor
    """
    # taxon labels must match for each phy and dat replicate
    phy_labels = set([ n.taxon.label for n in phy.leaf_nodes() ])
    dat_labels = set( dat.columns.to_list()[1:] )   # skip first element 'taxa'
    phy_missing = phy_labels.difference(dat_labels)

    if len(phy_missing) != 0:
        phy_missing = sorted(list(phy_missing))
        err_msg = f'Missing taxon labels in dat but not in phy for replicate {idx}: '
        err_msg += ' '.join(phy_missing)
        raise ValueError(err_msg)

    cpvs = None
    if tree_type == 'serial':
        cpvs = encode_cblvs(phy, dat, tree_width,
                            tree_encode_type, rescale)
    elif tree_type == 'extant':
        cpvs = encode_cdvs(phy, dat, tree_width,
                           tree_encode_type, rescale)
    else:
        ValueError(f'Unrecognized {tree_type}')

    return cpvs

def encode_cdvs(phy, dat, tree_width, tree_encode_type, rescale=True):
    """
    Encode Compact Diversity-reordered Vector + States (CDV+S) array

    # num columns equals tree_width, 0-padding
    # returns tensor with following rows
    # 0:  internal node root-distance
    # 1:  leaf node branch length
    # 2:  internal node branch length
    # 3+: state encoding

    Arguments:
        phy (dendropy.Tree):     phylogenetic tree
        dat (numpy.array):       character data
        tree_width (int):        number of columns (max. num. taxa)
                                 in CPVS array
        tree_encode_type (str):  type of tree encoding ('height_only' or
                                 'height_brlen')
        rescale:                 set tree height to 1 then encode, if True

    Returns:
        numpy.ndarray: The encoded CDV+S tensor.
    """

    # data dimensions
    num_tree_col = 0
    num_char_col = dat.shape[0]
    if tree_encode_type == 'height_only':
        num_tree_col = 1
    elif tree_encode_type == 'height_brlen':
        num_tree_col = 3

    # initialize workspace
    phy.calc_node_root_distances(return_leaf_distances_only=False)
    heights    = np.zeros( (tree_width, num_tree_col) )
    states     = np.zeros( (tree_width, num_char_col) )
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
            if tree_encode_type == 'height_brlen':
                heights[height_idx,1] = nd.edge.length
            states[state_idx,:]   = dat[nd.taxon.label].to_list()
            state_idx += 1
        else:
            heights[height_idx,0] = nd.root_distance
            if tree_encode_type == 'height_brlen':
                heights[height_idx,2] = nd.edge.length
            height_idx += 1

    # stack the phylo and states tensors
    if rescale:
        heights = heights / np.max(heights)
    phylo_tensor = np.hstack( [heights, states] )

    return phylo_tensor


def encode_cblvs(phy, dat, tree_width, tree_encode_type, rescale=True):
    """
    Encode Compact Bijective Ladderized Vector + States (CBLV+S) array

    # num columns equals tree_width, 0-padding
    # returns tensor with following rows
    # 0:  leaf node-to-last internal node distance
    # 1:  internal node root-distance
    # 2:  leaf node branch length
    # 3:  internal node branch length
    # 4+: state encoding

    Arguments:
        phy (dendropy.Tree):     phylogenetic tree
        dat (numpy.array):       character data
        tree_width (int):        number of columns (max. num. taxa)
                                 in CPVS array
        tree_encode_type (str):  type of tree encoding ('height_only' or
                                 'height_brlen')
        rescale:                 set tree height to 1 then encode, if True

    Returns:
        numpy.ndarray: The encoded CBLV+S tensor.
    """

    # data dimensions
    num_tree_col = 0
    num_char_col = dat.shape[0]
    if tree_encode_type == 'height_only':
        num_tree_col = 2
    elif tree_encode_type == 'height_brlen':
        num_tree_col = 4

    # initialize workspace
    phy.calc_node_root_distances(return_leaf_distances_only=False)
    heights    = np.zeros( (tree_width, num_tree_col) )
    states     = np.zeros( (tree_width, num_char_col) )
    state_idx  = 0
    height_idx = 0

    # postorder traversal to rotate nodes by max-root-distance
    for nd in phy.postorder_node_iter():
        if nd.is_leaf():
            nd.max_root_distance = nd.root_distance
        else:
            children                  = nd.child_nodes()
            ch_max_root_distance      = [ ch.max_root_distance for ch in children ]
            ch_max_root_distance_rank = np.argsort( ch_max_root_distance )[::-1]  # [0,1] or [1,0]
            children_reordered        = [ children[i] for i in ch_max_root_distance_rank ]
            nd.max_root_distance      = max(ch_max_root_distance)
            nd.set_children(children_reordered)

    # inorder traversal to fill matrix
    last_int_node = phy.seed_node
    last_int_node.edge.length = 0
    for nd in phy.inorder_node_iter():
        if nd.is_leaf():
            heights[height_idx,0] = nd.root_distance - last_int_node.root_distance
            if tree_encode_type == 'height_brlen':
                heights[height_idx,2] = nd.edge.length
            states[state_idx,:]   = dat[nd.taxon.label].to_list()
            state_idx += 1
        else:
            heights[height_idx+1,1] = nd.root_distance
            if tree_encode_type == 'height_brlen':
                heights[height_idx+1,3] = nd.edge.length
            last_int_node = nd
            height_idx += 1

    # stack the phylo and states tensors
    if rescale:
        heights = heights / np.max(heights)
    phylo_tensor = np.hstack( [heights, states] )

    return phylo_tensor


def make_prune_phy(phy, prune_fn, rel_extant_age_tol=1e-10):
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
    phy_ = phy  # copy.deepcopy(phy)
    # compute all root-to-node distances
    root_distances = phy_.calc_node_root_distances()
    # find tree height (max root-to-node distance)
    tree_height = np.max(root_distances)
    # tips are considered "at present" if age is within 0.0001 * tree_height
    tol = tree_height * rel_extant_age_tol
    # create empty dictionary
    d = {}
    # loop through all leaf nodes
    leaf_nodes = phy_.leaf_nodes()
    for i, nd in enumerate(leaf_nodes):
        # convert root-distances to ages
        age = tree_height - nd.root_distance
        nd.annotations.add_new('age', age)
        # ultrametricize ages for extant taxa
        if age < tol:
            age = 0.0
        # store taxon and age in dictionary
        taxon_name = str(nd.taxon).strip('\'')
        taxon_name = taxon_name.replace(' ', '_')
        d[taxon_name] = age
    # determine what to drop
    drop_taxon_labels = [k for k, v in d.items() if v > tol ]
    # inform user if pruning yields valid tree
    if len(leaf_nodes) - len(drop_taxon_labels) < 2:
        return None
    else:
        # prune non-extant taxa
        phy_.prune_taxa_with_labels(drop_taxon_labels)
        phy_.purge_taxon_namespace()
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
    phy_ = phy    # copy.deepcopy(phy)

    # get number of taxa
    leaf_nodes = phy_.leaf_nodes()
    num_taxa = len(leaf_nodes)

    # if downsampling is needed
    if num_taxa > max_taxa:

        # MJL: note, Format bottleneck with large trees. It should be possible
        #      to write faster code when we don't care about taxon labels or
        #      downstream use of the dendropy object?
        # https://dendropy.org/_modules/dendropy/datamodel/treemodel/_tree#Tree.prune_taxa_with_labels

        # shuffle taxa indices
        rand_idx = list(range(num_taxa))
        np.random.shuffle(rand_idx)
        
        # clean old namespace
        phy_.purge_taxon_namespace()

        # get all taxa beyond max_taxa threshold
        drop_taxa = []
        for i in rand_idx[max_taxa:]:
            drop_taxa.append(phy_.taxon_namespace[i])

        # drop those taxa
        phy_.prune_taxa(drop_taxa)

        # verify resultant tree size
        assert (len(phy_.leaf_nodes()) == max_taxa)

    # save downsampled tree
    phy_.write(path=down_fn, schema='newick')

    # done
    return phy_


# make matrix with parameter values, lower-bounds, upper-bounds: 3D->2D
def make_param_VLU_mtx(A, param_names):
    """Make parameter Value-Lower-Upper matrix.

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
    index = pd.MultiIndex.from_product([range(s) for s in A.shape],
                                       names=['stat', 'rep_idx', 'param'])

    # flattened data frame
    df = pd.DataFrame({'A': A.flatten()}, index=index)['A']
    df = df.reorder_levels(['param', 'stat', 'rep_idx']).sort_index()

    # unstack stat and param, so they become combined header indices
    df = df.unstack(level=['stat', 'param'])
    # col_names = df.columns

    new_col_names = [f'{param_names[y]}_{stat_names[x]}' for x, y in df.columns]
    df.columns = new_col_names

    return df


def make_clean_phyenc_str(x):
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

    def numpy_formatter(y):
        if y % 1 == 0:
            return "{:d}".format(int(y))
        else:
            return NUMPY_FLOAT_FMT_STR.format(y)

    s = np.array2string(x, separator=',', max_line_width=1e200,
                        threshold=1e200, edgeitems=1e200,
                        floatmode='maxprec',
                        formatter={'float_kind': numpy_formatter})

    s = re.sub(r'[\[\]]', '', string=s)
    s = re.sub(r',\n ', '\n', string=s)

    return s


def ndarray_to_flat_str(x):
    """Converts a numpy.ndarray into flattend csv vector."""

    # numpy formatter for floats & ints
    def numpy_formatter(y):
        if y % 1 == 0:
            return "{:d}".format(int(y))
        else:
            return NUMPY_FLOAT_FMT_STR.format(y)

    # convert ndarray to formatted string
    s = np.array2string(x, separator=',', max_line_width=1e200,
                        threshold=1e200, edgeitems=1e200,
                        floatmode='maxprec',
                        formatter={'float_kind': numpy_formatter})
    # remove brackets, whitespace
    s = re.sub(r'[\[\]\n ]', '', string=s)
    # endline
    # s = s + '\n'
    return s


##################################################

#########################
# Tensor de/normalizing #
#########################

def safe_log_tensor(data, col, log_offset=0.0):
    assert col is list()
    assert np.all([type(x) is bool for x in col])
    assert np.all(data[:, col] >= 0.0)

    # y = log(x + c)
    data[:, col] = np.log(data[:, col] + log_offset)

    return data


def safe_delog_tensor(data, col, log_offset=0.0):
    assert col is list()
    assert np.all([type(x) is bool for x in col])

    # x = exp(y) - c
    data[:, col] = np.exp(data[:, col]) - log_offset
    assert np.all(data[:, col] >= 0.0)

    return data


def normalize(data, m_sd=None):
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
    if type(m_sd) is type(None):
        m = data.mean(axis=0)
        sd = data.std(axis=0)
        sd[np.where(sd == 0)] = 1
        return (data - m) / sd, m, sd
    else:
        m_sd[1][np.where(m_sd[1] == 0)] = 1
        return (data - m_sd[0]) / m_sd[1]


def denormalize(data, m_sd, exp=False, tol=300):
    """
    Denormalize the data using the mean and standard deviation.

    This function denormalizes the input data using the provided mean and
    standard deviation. It reverses the normalization process and brings the
    data back to its original scale.

    Args:
        data (numpy.ndarray): The normalized data to be denormalized.
        m_sd (tuple): A tuple containing the mean and standard deviation used
            for normalization.
        exp (bool): If True, the data is exponentiated after denormalization.
        tol (float): The tolerance value used to clip the denormalized data.

    Returns:
        numpy.ndarray: The denormalized data.
    """
    if exp:
        x = data * m_sd[1] + m_sd[0]
        x[x >= tol] = tol
        x[x <= -tol] = -tol
        return np.exp(x)
    else:
        return data * m_sd[1] + m_sd[0]


##################################################

######################
# phyddle print info #
######################

def phyddle_str(s, style=1, color=34):
    """Make printable phyddle output string.
    
    Args:
        s (str): The string to be styled.
        style (int, optional): Style code.
        color (int, optional): Foreground color code.
    
    Returns:
        str: The styled string.

    """

    c_start = f'\x1b[{style};{color};m'
    c_end = '\x1b[0m'
    x = c_start + s + c_end
    return x


def print_str(s, verbose=True, style=1, color=34):
    """Prints a phyddle string to the standard output.

    Args:
        s (str): The string to be styled.
        style (int, optional): Style code.
        color (int, optional): Foreground color code.
        verbose (bool, optional): Prints the formatted string if True.

    """

    if verbose:
        print(phyddle_str(s, style, color))
    return


def print_err(s, verbose=True, style=1, color=34, exit=False):
    """Prints a phyddle error to the standard output.

    Args:
        s (str): The string to be styled.
        style (int, optional): Style code.
        color (int, optional): Foreground color code.
        verbose (bool, optional): Prints the formatted string if True.

    """
    print_str(f'ERROR: {s}', verbose, style, color)
    if exit:
        sys.exit()
    return


def print_warn(s, verbose=True, style=1, color=34):
    """Prints a phyddle warning to the standard output.

    Args:
        s (str): The string to be styled.
        style (int, optional): Style code.
        color (int, optional): Foreground color code.
        verbose (bool, optional): Prints the formatted string if True.

    """

    print_str(f'WARNING: {s}', verbose, style, color)
    return


def phyddle_header(s, style=1, color=34):
    """Generate a phyddle header string.
    
    Args:
        s (str): The string to be styled.
        style (int, optional): Style code.
        color (int, optional): Foreground color code.

    Returns:
        str: The header string.

    """

    version = f'v{PHYDDLE_VERSION}'.rjust(8, ' ')

    steps = {'sim': 'Simulating',
             'fmt': 'Formatting',
             'trn': 'Training',
             'est': 'Estimating',
             'plt': 'Plotting'}

    x = ''

    if s == 'title':
        x = phyddle_str('┏━━━━━━━━━━━━━━━━━━━━━━┓', style, color) + '\n'
        x += phyddle_str(f'┃   phyddle {version}   ┃', style, color) + '\n'
        x += phyddle_str('┣━━━━━━━━━━━━━━━━━━━━━━┫', style, color)

    elif s in list(steps.keys()):
        step_name = steps[s] + '...'
        step_name = step_name.ljust(13, ' ')
        x = phyddle_str('┃                      ┃', style, color) + '\n'
        x += phyddle_str(f'┗━┳━▪ {step_name} ▪━━┛', style, color)

    return x


def print_step_header(step, in_dir, out_dir, in_prefix, out_prefix,
                      verbose=True, style=1, color=34):
    """Generate a phyddle step info string.
    
    Args:
        step (str): The step symbol.
        in_dir (list): A list of input directories.
        out_dir (str): The output directory.
        in_prefix (list): A list of input prefixes.
        out_prefix (str): The output prefix.
        verbose (bool, optional): Prints the formatted string if True.
        style (int, optional): Style code.
        color (int, optional): Foreground color code.
        
    Returns:
        str: The information string.

    """

    # header
    run_info = phyddle_header(step) + '\n'

    # get ljust
    num_ljust = len(out_dir)
    if in_dir is not None:
        num_ljust = max([len(x) for x in in_dir + [out_dir]])

    # in paths
    plot_bar = True
    if in_dir is not None:
        run_info += phyddle_str('  ┃') + '\n'
        plot_bar = False
        for i, _in_dir in enumerate(in_dir):
            in_pfx = in_prefix[i]
            in_path = f'{_in_dir}'.ljust(num_ljust, ' ')
            if in_pfx != 'out':
                in_path += f'  [prefix: {in_pfx}]'
            if i == 0:
                run_info += phyddle_str(f'  ┣━━━▪ input:   {in_path}', style,
                                        color) + '\n'
            else:
                run_info += phyddle_str(f'  ┃              {in_path}', style,
                                        color) + '\n'

    # out path
    out_path = ''
    if out_dir is not None:
        # run_info += phyddle_str('  ┃')  + '\n'
        out_path = f'{out_dir}'.ljust(num_ljust, ' ')
        if out_prefix != 'out':
            out_path += f'  [prefix: {out_prefix}]'

    if plot_bar:
        run_info += phyddle_str('  ┃') + '\n'

    run_info += phyddle_str(f'  ┗━━━▪ output:  {out_path}') + '\n'

    # print if verbose is True
    if verbose:
        print(run_info)

    return


##################################################

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
        self.args = args
        self.arg_str = self.make_arg_str()
        self.job_id = self.args['job_id']
        # self.work_dir    = self.args['work_dir']
        self.log_dir = self.args['log_dir']
        self.date_str = self.args['date']
        # self.proj        = self.args['proj']

        # collect other info and set constants
        self.pkg_name = 'phyddle'
        self.version = PHYDDLE_VERSION  # pkg_resources.get_distribution(self.pkg_name).version
        self.commit = '(to be done)'
        self.command = ' '.join(sys.argv)
        self.max_lines = 1e5

        # filesystem
        self.base_fn = f'{self.pkg_name}_{self.version}_{self.date_str}'
        # self.base_dir    = f'{self.work_dir}/{self.proj}/{self.log_dir}'
        self.base_dir = f'{self.log_dir}'
        self.base_fp = f'{self.base_dir}/{self.base_fn}'
        self.fn_dict = {
            'run': f'{self.base_fp}.run.log',
            'sim': f'{self.base_fp}.simulate.log',
            'fmt': f'{self.base_fp}.format.log',
            'trn': f'{self.base_fp}.train.log',
            'est': f'{self.base_fp}.estimate.log',
            'plt': f'{self.base_fp}.plot.log'
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
        for k, v in self.args.items():
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

        assert (step in self.fn_dict.keys())
        fn = self.fn_dict[step]
        with open(fn, 'a') as file:
            file.write(f'{msg}\n')
        return

    def save_run_log(self):
        """Saves run log file."""
        fn = self.fn_dict['run']
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
        # s +=  'commit = TBD\n'
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
        installed_packages = {d.key for d in pkg_resources.working_set}
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
        s += 'Python packages:\n'
        for k, v in d.items():
            s += f'{k}\t{v}\n'

        return s

##################################################

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
