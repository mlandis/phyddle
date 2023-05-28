#!/usr/local/bin/python3

###################
# PHYDDLE IMPORTS #
###################

import ModelLoader
import Predicting
import Encoding
import Utilities

########################
# LOAD PIPELINE CONFIG #
########################

my_args = Utilities.load_config('config', arg_overwrite=True)

#########################
# DEFINE PIPELINE STEPS #
#########################

MyEncoder = Encoding.Encoder
my_mdl = ModelLoader.load_model(my_args)

MyEncoder = Encoding.Encoder
my_enc = MyEncoder(my_args, my_mdl)

pred_prefix = my_args['pred_dir'] + '/' + my_args['pred_prefix']
print(pred_prefix)
my_enc.encode_one(tmp_fn=pred_prefix, idx=-1)

MyPredictor = Predicting.Predictor
my_prd = MyPredictor(my_args)

################
# RUN PIPELINE #
################

my_prd.run()
