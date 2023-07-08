#!/usr/local/bin/python3

###################
# PHYDDLE IMPORTS #
###################

from phyddle import ModelLoader
from phyddle import Formatting
from phyddle import Utilities

########################
# LOAD PIPELINE CONFIG #
########################

my_args = Utilities.load_config('config', arg_overwrite=True)


#########################
# DEFINE PIPELINE STEPS #
#########################

# formatter prepares tensor format
#MyFormatter = Formatting.Formatter
#my_mdl = ModelLoader.load(my_args)
#my_fmt = MyFormatter(my_args, my_mdl)
my_fmt = Formatting.load(my_args) #, my_mdl)


################
# RUN PIPELINE #
################

# Step 1: format and encode output
my_fmt.run()
