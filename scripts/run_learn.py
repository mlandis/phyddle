#!/usr/local/bin/python3

###################
# PHYDDLE IMPORTS #
###################

from phyddle import Learning
from phyddle import Utilities

########################
# LOAD PIPELINE CONFIG #
########################

my_args = Utilities.load_config('config', arg_overwrite=True)


#########################
# DEFINE PIPELINE STEPS #
#########################

# trainer fits neural network
#MyLearner = Learning.CnnLearner
#my_lrn = MyLearner(my_args)
my_lrn = Learning.load(my_args)


################
# RUN PIPELINE #
################

# Step 1: train network
my_lrn.run()

