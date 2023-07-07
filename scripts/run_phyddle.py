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
my_mdl = ModelLoader.load_model(my_args)

# simulator samples from model
my_sim = Simulating.load_simulator(my_args, my_mdl)

# formatter prepares entire dataset into single tensor
MyFormatter = Formatting.Formatter
my_fmt = MyFormatter(my_args, my_mdl)

# learner fits neural network
MyLearner = Learning.CnnLearner
my_lrn = MyLearner(my_args)

# predicter uses trained network for estimates on new dataset
MyPredictor = Predicting.Predictor
my_prd = MyPredictor(my_args)

# plotter generates figures
MyPlotter = Plotting.Plotter
my_plt = MyPlotter(my_args)


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

