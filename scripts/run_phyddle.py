#!/usr/local/bin/python3

###################
# PHYDDLE IMPORTS #
###################

from phyddle import ModelLoader
from phyddle import Simulating
from phyddle import Formatting
from phyddle import Learning
from phyddle import Predicting
from phyddle import Plotting
from phyddle import Utilities

########################
# LOAD PIPELINE CONFIG #
########################

my_args = Utilities.load_config('config', arg_overwrite=True)

#########################
# DEFINE PIPELINE STEPS #
#########################

# create model template
my_mdl = ModelLoader.load(my_args)

# simulator samples from model
my_sim = Simulating.load(my_args, my_mdl)

# formatter prepares entire dataset into single tensor
#MyFormatter = Formatting.Formatter
my_fmt = Formatting.load(my_args) #, my_mdl)

# learner fits neural network
my_lrn = Learning.load(my_args)

# predicter uses trained network for estimates on new dataset
my_prd = Predicting.load(my_args)

# plotter generates figures
my_plt = Plotting.load(my_args)


################
# RUN PIPELINE #
################

# Step 1: run simulation
my_sim.run()

# Step 2: make tensor from output
my_fmt.run()

# Step 3: train network
my_lrn.run()

# Step 4: predict against new test dataset
pred_prefix = f"{my_args['pred_dir']}/{my_args['proj']}/{my_args['pred_prefix']}"
my_fmt.encode_one(tmp_fn=pred_prefix, idx=-1, save_phyenc_csv=True)
my_prd.run()

# Step 5: plot results
my_plt.run()

