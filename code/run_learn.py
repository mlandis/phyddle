#!/usr/local/bin/python3

###################
# PHYDDLE IMPORTS #
###################

import Learning
import Utilities

########################
# LOAD PIPELINE CONFIG #
########################

my_args = Utilities.load_config('config', arg_overwrite=True)

#my_all_args = my_config.my_all_args
#my_lrn_args = my_config.my_lrn_args


#########################
# DEFINE PIPELINE STEPS #
#########################

# trainer fits neural network
MyLearner = Learning.CnnLearner
my_lrn = MyLearner(my_args)


################
# RUN PIPELINE #
################

# Step 3: train network
my_lrn.run()

