#!/usr/local/bin/python3

###################
# PHYDDLE IMPORTS #
###################

#import Model
import ModelLoader
import Simulator
import Utilities

########################
# LOAD PIPELINE CONFIG #
########################

my_config = Utilities.load_config('config', arg_overwrite=True)

my_all_args = my_config.my_all_args
my_mdl_args = my_config.my_mdl_args
my_sim_args = my_config.my_sim_args


#########################
# DEFINE PIPELINE STEPS #
#########################

# simulator samples from model
MySimulator = Simulator.MasterSimulator
my_mdl = ModelLoader.load_model(my_mdl_args)
my_sim = MySimulator(my_sim_args, my_mdl)


################
# RUN PIPELINE #
################

# Step 1: run simulation
my_sim.run()
