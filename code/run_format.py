#!/usr/local/bin/python3

###################
# PHYDDLE IMPORTS #
###################

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
MyInputFormatter = Formatting.InputFormatter
my_fmt = MyInputFormatter(my_args)


################
# RUN PIPELINE #
################

# Step 2: re-format output
my_fmt.run()
