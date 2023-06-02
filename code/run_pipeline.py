#!/usr/local/bin/python3

###################
# PHYDDLE IMPORTS #
###################

import ModelLoader
import Simulating
#import Encoding
import Formatting
import Learning
import Plotting
import Utilities

########################
# LOAD PIPELINE CONFIG #
########################

my_args = Utilities.load_config('config', arg_overwrite=True)

#########################
# DEFINE PIPELINE STEPS #
#########################

# simulator samples from model
MySimulator = Simulating.MasterSimulator
my_mdl = ModelLoader.load_model(my_args)
my_sim = MySimulator(my_args, my_mdl)

# encoder converts raw data into matrices
#MyEncoder = Encoding.Encoder
#my_enc = MyEncoder(my_args, my_mdl)

# formatter prepares entire dataset into single tensor
MyFormatter = Formatting.Formatter
my_fmt = MyFormatter(my_args, my_mdl)

# trainer fits neural network
MyLearner = Learning.CnnLearner
my_lrn = MyLearner(my_args)

# plotter generates figures
MyPlotter = Plotting.Plotter
my_plt = MyPlotter(my_args)

################
# RUN PIPELINE #
################

# Step 1: run simulation
my_sim.run()

# Step 2: encode output
# my_enc.run()

# Step 3: make tensor from output
my_fmt.run()

# Step 4: train network
my_lrn.run()

# Step 5: plot results
my_plt.run()

# Step 6: predict against new test dataset
# my_pred.run()
