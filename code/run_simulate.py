#!/usr/local/bin/python3

###################
# PHYDDLE IMPORTS #
###################

import ModelLoader
import Simulating
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


################
# RUN PIPELINE #
################

# Step 1: run simulation
my_sim.run()

