#!/usr/local/bin/python3

###################
# PHYDDLE IMPORTS #
###################

#import Model
#import ModelLoader
#import Simulator
import Plotting
import Utilities

########################
# LOAD PIPELINE CONFIG #
########################

my_args = Utilities.load_config('config', arg_overwrite=True)

# my_all_args = my_config.my_all_args
# my_mdl_args = my_config.my_mdl_args
# my_plt_args = my_config.my_plt_args


#########################
# DEFINE PIPELINE STEPS #
#########################

# encoder converts raw output into encoded output
MyPlotter = Plotting.Plotter
#my_mdl = ModelLoader.load_model(my_mmy_argdl_args)
my_plt = MyPlotter(my_args)


################
# RUN PIPELINE #
################

# Step 1: run simulation
my_plt.run()

