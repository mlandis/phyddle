#!/usr/local/bin/python3

###################
# PHYDDLE IMPORTS #
###################

from phyddle import Plotting
from phyddle import Utilities

########################
# LOAD PIPELINE CONFIG #
########################

my_args = Utilities.load_config('config', arg_overwrite=True)


#########################
# DEFINE PIPELINE STEPS #
#########################

# plotter generates figures
#MyPlotter = Plotting.Plotter
#my_plt = MyPlotter(my_args)
my_plt = Plotting.load(my_args)


################
# RUN PIPELINE #
################

# Step 1: run plotter
my_plt.run()

