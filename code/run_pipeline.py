#!/usr/bin/python3
import scipy as sp
import numpy as np
import phyddle_util
import model
import simulator
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
    'rep_idx'           : list(range(0, 5)),
    'tree_sizes'        : [ 200, 500 ],
    'use_parallel'      : False,
    'start_sizes'       : {},                # move to mode; none, default 0
    'start_state'       : { 'S' : 0 },       # move to model
    'sample_population' : ['S'],
    'stop_floor_sizes'  : 0,
    'stop_ceil_sizes'   : 300                # something weird about MASTER stop rule?
} | my_all_args


# define tensor-formatting settings
my_fmt_args = {
    'fmt_dir' : 'tensor_data'
} | my_all_args

# define training settings
my_trn_args = { } | my_all_args
# define result-making settings
my_res_args = { } | my_all_args

# define pipeline steps
# (replace classes as desired)
MySimulator = simulator.MasterSimulator
my_mdl = model.make_model(my_mdl_args)
my_sim = MySimulator(my_sim_args, my_mdl)
my_sim.run()

#MyFormatter = Formatter
#MyTrainer   = CNNTrainer
#MyResulter  = Resulter

# define pipeline objects
#my_mdl = MyModel(**my_mdl_args)

#my_fmt = MyFormatter(my_fmt_args)
#my_trn = MyTrainer(my_trn_args)
#my_res = MyResulter(my_res_args)

# run pipeline steps

#my_fmt.run()
#my_trn.run()
#my_res.run()