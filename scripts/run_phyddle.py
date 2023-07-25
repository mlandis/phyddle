#!/usr/bin/env python3

########################
# LOAD PIPELINE CONFIG #
########################

from phyddle import utilities

my_args = utilities.load_config('config', arg_overwrite=True)
step = my_args['step']

logger = utilities.Logger(my_args)
logger.save_log('run')

################
# RUN PIPELINE #
################

# Step 1: simulate training data
if 'S' in step:
    from phyddle import ModelLoader
    from phyddle import simulate
    my_mdl = ModelLoader.load(my_args)
    my_sim = simulate.load(my_args, my_mdl)
    my_sim.run()

# Step 2: format training data into tensors
if 'F' in step:
    from phyddle import Formatting
    my_fmt = Formatting.load(my_args)
    my_fmt.run()

# Step 3: train network with training data
if 'T' in step:
    from phyddle import Learning
    my_lrn = Learning.load(my_args)
    my_lrn.run()

# Step 4: estimates for new dataset
if 'E' in step:
    prd_prefix = f"{my_args['prd_dir']}/{my_args['proj']}/{my_args['prd_prefix']}"
    from phyddle import Formatting
    from phyddle import Predicting
    my_fmt = Formatting.load(my_args)
    my_fmt.encode_one(tmp_fn=prd_prefix, idx=-1, save_phyenc_csv=True)
    my_prd = Predicting.load(my_args)
    my_prd.run()

# Step 5: plot results
if 'P' in step:
    from phyddle import Plotting
    my_plt = Plotting.load(my_args)
    my_plt.run()

