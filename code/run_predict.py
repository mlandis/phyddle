#!/usr/local/bin/python3

###################
# PHYDDLE IMPORTS #
###################

import ModelLoader
import Predicting
import Formatting
import Utilities


########################
# LOAD PIPELINE CONFIG #
########################

my_args = Utilities.load_config('config', arg_overwrite=True)


#########################
# DEFINE PIPELINE STEPS #
#########################

MyFormatter = Formatting.Formatter
my_mdl = ModelLoader.load_model(my_args)
my_fmt = MyFormatter(my_args, my_mdl)

pred_prefix = my_args['pred_dir'] + '/' + my_args['pred_prefix']
my_fmt.encode_one(tmp_fn=pred_prefix, idx=-1, save_phyvec=True)

MyPredictor = Predicting.Predictor
my_prd = MyPredictor(my_args)


################
# RUN PIPELINE #
################

my_prd.run()
