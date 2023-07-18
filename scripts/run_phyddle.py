#!/usr/local/bin/python3

########################
# LOAD PIPELINE CONFIG #
########################

from phyddle import Utilities
from phyddle import Logging

my_args = Utilities.load_config('config', arg_overwrite=True)
step = my_args['step']

logger = Logging.Logger(my_args)
logger.save_log('run')

################
# RUN PIPELINE #
################

# Step 1: simulate training data
if 'sim' in step:
    from phyddle import ModelLoader
    from phyddle import Simulating
    my_mdl = ModelLoader.load(my_args)
    my_sim = Simulating.load(my_args, my_mdl)
    my_sim.run()

# Step 2: format training data into tensors
if 'fmt' in step:
    from phyddle import Formatting
    my_fmt = Formatting.load(my_args)
    my_fmt.run()

# Step 3: train network with training data
if 'lrn' in step:
    from phyddle import Learning
    my_lrn = Learning.load(my_args)
    my_lrn.run()

# Step 4: predict for new dataset
if 'prd' in step:
    pred_prefix = f"{my_args['pred_dir']}/{my_args['proj']}/{my_args['pred_prefix']}"
    from phyddle import Formatting
    from phyddle import Predicting
    my_fmt = Formatting.load(my_args)
    my_fmt.encode_one(tmp_fn=pred_prefix, idx=-1, save_phyenc_csv=True)
    my_prd = Predicting.load(my_args)
    my_prd.run()

# Step 5: plot results
if 'plt' in step:
    from phyddle import Plotting
    my_plt = Plotting.load(my_args)
    my_plt.run()

