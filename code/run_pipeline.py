#!/usr/local/bin/python3

###################
# PHYDDLE IMPORTS #
###################

import Model
import Simulator
import InputFormatter
import Learner
import Utilities

########################
# LOAD PIPELINE CONFIG #
########################

my_config = Utilities.load_config('config', arg_overwrite=True)

my_all_args = my_config.my_all_args
my_mdl_args = my_config.my_mdl_args
my_sim_args = my_config.my_sim_args
my_fmt_args = my_config.my_fmt_args
my_lrn_args = my_config.my_lrn_args


#########################
# DEFINE PIPELINE STEPS #
#########################

# simulator samples from model
MySimulator = Simulator.MasterSimulator
#MySimulator = Simulator.PhyloJunction
my_mdl = Model.make_model(my_mdl_args)
my_sim = MySimulator(my_sim_args, my_mdl)

# formatter prepares tensor format
MyInputFormatter = InputFormatter.InputFormatter
my_fmt = MyInputFormatter(my_fmt_args)

# trainer fits neural network
MyLearner = Learner.CnnLearner
my_lrn = MyLearner(my_lrn_args)


################
# RUN PIPELINE #
################

# Step 1: run simulation
my_sim.run()

# Step 2: re-format output
my_fmt.run()

# Step 3: train network
my_lrn.run()

# Step 4: separate plotting functionality?
# possibly better to have Learner save output tp file, then have Plotter generate figures
# my_plt.run()
