#!/usr/local/bin/python3

###################
# PHYDDLE IMPORTS #
###################

from phyddle import ModelLoader
from phyddle import Simulating
from phyddle import Utilities

########################
# LOAD PIPELINE CONFIG #
########################

my_args = Utilities.load_config('config', arg_overwrite=True)


#########################
# DEFINE PIPELINE STEPS #
#########################

# simulator samples from model
my_mdl = ModelLoader.load(my_args)
my_sim = Simulating.load(my_args, my_mdl) 


################
# RUN PIPELINE #
################

# Step 1: run simulation
my_sim.run()

