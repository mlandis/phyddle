#!/usr/bin/env python3

########################
# LOAD PIPELINE CONFIG #
########################

import phyddle
import phyddle.utilities as util

my_args = util.load_config('config.py', arg_overwrite=True)
step = my_args['step']

logger = util.Logger(my_args)
logger.save_log('run')

################
# RUN PIPELINE #
################

#import phyddle.model_loader as mdl_ldr
import phyddle.simulate as sim
import phyddle.format as fmt
import phyddle.train as trn
import phyddle.estimate as est
import phyddle.plot as plt

# Step 1: simulate training data
if 'S' in step:
    #my_mdl = mdl_ldr.load(my_args)
    my_sim = sim.load(my_args) #, my_mdl)
    my_sim.run()

# Step 2: format training data into tensors
if 'F' in step:
    my_fmt = fmt.load(my_args)
    my_fmt.run()

# Step 3: train network with training data
if 'T' in step:
    my_trn = trn.load(my_args)
    my_trn.run()

# Step 4: estimates for new dataset
if 'E' in step:
    est_prefix = f"{my_args['est_dir']}/{my_args['proj']}/{my_args['est_prefix']}"
    my_fmt = fmt.load(my_args)
    my_fmt.encode_one(tmp_fn=est_prefix, idx=-1, save_phyenc_csv=True)
    my_est = est.load(my_args)
    my_est.run()

# Step 5: plot results
if 'P' in step:
    my_plt = plt.load(my_args)
    my_plt.run()

