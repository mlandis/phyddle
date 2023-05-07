#!/usr/bin/python3
import scipy as sp
import numpy as np
#import phyddle_util
import model
import Simulator
import InputFormatter
import Learner
#import formatter
#import trainer
#import resulter

#####################
# PIPELINE SETTINGS #
#####################
my_all_args = { 'job_name' : 'test' }

##################
# MODEL SETTINGS #
##################
my_mdl_args = {
    'model_type'    : 'geosse',
    'model_variant' : 'equal_rates',
    'num_locations' : 3,
    'rv_fn' : {
        'w': sp.stats.expon.rvs,
        'e': sp.stats.expon.rvs,
        'd': sp.stats.expon.rvs,
        'b': sp.stats.expon.rvs },
    'rv_arg' : {
        'w': { 'scale' : 1.0 },
        'e': { 'scale' : 0.5 },
        'd': { 'scale' : 1.0 },
        'b': { 'scale' : 3.0 }
    }
}

#######################
# SIMULATION SETTINGS #
#######################
my_sim_args = {
    'sim_dir'           : '../raw_data',
    'rep_idx'           : list(range(0, 500)),
    'tree_sizes'        : [ 200, 500 ],
    'use_parallel'      : True,
    'start_sizes'       : {},                # move to mode; none, default 0
    'start_state'       : { 'S' : 0 },       # move to model
    'sample_population' : ['S'],
    'stop_floor_sizes'  : 0,
    'stop_ceil_sizes'   : 300                # something weird about MASTER stop rule?
} | my_all_args


# define tensor-formatting settings
my_fmt_args = {
    'fmt_dir' : '../tensor_data',
    'sim_dir' : '../raw_data'
} | my_all_args

# define learning settings
my_lrn_args = { 
    'fmt_dir'        : '../tensor_data',
    'net_dir'        : '../network',
    'plt_dir'        : '../plot',
    'tree_size'      : 500,
    'tree_type'      : 'extant',
    'predict_idx'    : [ 0, 3, 6, 18 ],
    'num_epochs'     : 5,
    'num_test'       : 5,
    'num_validation' : 5,
    'batch_size'     : 8,
    'loss'           : 'mae',
    'optimizer'      : 'adam',
    'metrics'        : ['mae', 'acc', 'mape']
} | my_all_args

# define plot & results-making settings
my_plt_args = { 
    'net_dir'       : '../plot'
} | my_all_args

#########################
# DEFINE PIPELINE STEPS #
#########################

# simulator samples from model
MySimulator = Simulator.MasterSimulator
my_mdl = model.make_model(my_mdl_args)
my_sim = MySimulator(my_sim_args, my_mdl)

# formatter prepares tensor format
MyInputFormatter = InputFormatter.InputFormatter
my_fmt = MyInputFormatter(my_fmt_args)

# trainer fits neural network
MyLearner = Learner.CnnLearner
my_lrn = MyLearner(my_lrn_args)


# plotter generates output
#MyPlotter = Plotter
#my_plt = MyPlotter(my_plt_args)

################
# RUN PIPELINE #
################

#my_sim.run()
#my_fmt.run()
my_lrn.run()
#my_plt.run()
