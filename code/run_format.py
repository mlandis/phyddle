#!/usr/local/bin/python3

###################
# PHYDDLE IMPORTS #
###################

import Model
import InputFormatter
import Utilities

########################
# LOAD PIPELINE CONFIG #
########################

my_config = Utilities.load_config('phyddle_config', arg_overwrite=True)

my_all_args = my_config.my_all_args
my_fmt_args = my_config.my_fmt_args


#########################
# DEFINE PIPELINE STEPS #
#########################

# formatter prepares tensor format
MyInputFormatter = InputFormatter.InputFormatter
my_fmt = MyInputFormatter(my_fmt_args)


################
# RUN PIPELINE #
################

# Step 2: re-format output
my_fmt.run()
