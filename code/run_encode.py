#!/usr/local/bin/python3

###################
# PHYDDLE IMPORTS #
###################

#import Model
import ModelLoader
#import Simulator
import Encoding
import Utilities

########################
# LOAD PIPELINE CONFIG #
########################

my_args = Utilities.load_config('config', arg_overwrite=True)

# my_all_args = my_config.my_all_args
# my_mdl_args = my_config.my_mdl_args
# my_sim_args = my_config.my_sim_args


#########################
# DEFINE PIPELINE STEPS #
#########################

# encoder converts raw output into encoded output
MyEncoder = Encoding.Encoder
my_mdl = ModelLoader.load_model(my_args)
my_enc = MyEncoder(my_args, my_mdl)


################
# RUN PIPELINE #
################

# Step 1: run simulation
my_enc.run()

