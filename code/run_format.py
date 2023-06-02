#!/usr/local/bin/python3

###################
# PHYDDLE IMPORTS #
###################

import ModelLoader
import Formatting
import Utilities

########################
# LOAD PIPELINE CONFIG #
########################

my_args = Utilities.load_config('config', arg_overwrite=True)

# my_all_args = my_config.my_all_args
# my_fmt_args = my_config.my_fmt_args


#########################
# DEFINE PIPELINE STEPS #
#########################

# formatter prepares tensor format
MyFormatter = Formatting.Formatter
my_mdl = ModelLoader.load_model(my_args)
my_fmt = MyFormatter(my_args, my_mdl)


################
# RUN PIPELINE #
################

# Step 2: re-format output
my_fmt.run()
