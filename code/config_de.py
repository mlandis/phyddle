import scipy as sp

# helper variables
num_char = 3

#####################
# PIPELINE SETTINGS #
#####################
my_all_args = { 'job_name' : 'geosse_de_1' }

##################
# MODEL SETTINGS #
##################
my_mdl_args = {
    'model_type'    : 'geosse',
    'model_variant' : 'density_effect',
    'num_char'      : num_char,
    'rv_fn' : {
        'w': sp.stats.expon.rvs,
        'e': sp.stats.expon.rvs,
        'd': sp.stats.expon.rvs,
        'b': sp.stats.expon.rvs,
        'ed': sp.stats.expon.rvs },
    'rv_arg' : {
        'w':  { 'scale' : 0.25 },
        'e':  { 'scale' : 0.05 },
        'd':  { 'scale' : 0.1 },
        'b':  { 'scale' : 0.5 },
        'ed': { 'scale' : 0.15 }
    }
}
my_all_args = my_all_args | my_mdl_args

######################
# SIMULATOR SETTINGS #
######################
my_sim_args = {
    'sim_dir'           : '../raw_data',
    'sim_logging'       : 'verbose',
    'start_idx'         : 0,
    'end_idx'           : 100,
    'tree_sizes'        : [ 200, 500 ],
    'use_parallel'      : True,
    'num_proc'          : -2,
    'sample_population' : ['S'],
    'stop_time'         : 10,
    'stop_floor_sizes'  : 0,
    'stop_ceil_sizes'   : 400                # MASTER seems to generate too many taxa?
} #| my_all_args
my_all_args = my_all_args | my_sim_args


####################
# ENCODER SETTINGS #
####################
my_enc_args = {

}
my_all_args = my_all_args | my_enc_args


#############################
# TENSOR-FORMATTER SETTINGS #
#############################
my_fmt_args = {
    'fmt_dir' : '../tensor_data',
    'tree_type'  : 'extant',
    'param_pred' : ['w_0', 'e_0', 'd_0_1', 'b_0_1', 'ed_0'],
    'param_data' : [],
    'tensor_format' : 'hdf5'
} #| my_all_args
my_all_args = my_all_args | my_fmt_args

#####################
# LEARNING SETTINGS #
#####################
my_lrn_args = { 
    'net_dir'        : '../network',
    'tree_size'      : 200,
    'num_epochs'     : 20,
    'prop_test'        : 0.05,
    'prop_validation'  : 0.05,
    'prop_calibration' : 0.20,
    'alpha_CQRI'     : 0.95,
    'batch_size'     : 128,
    'loss'           : 'mse',
    'optimizer'      : 'adam',
    'metrics'        : ['mae', 'acc', 'mape']
}
my_all_args = my_all_args | my_lrn_args


#####################
# PLOTTING SETTINGS #
#####################
my_plt_args = {
    'plt_dir'        : '../plot',
    'network_prefix' : 'sim_batchsize128_numepoch20_nt200'
}
my_all_args = my_all_args | my_plt_args


#######################
# PREDICTING SETTINGS #
#######################

my_prd_args = {
    'pred_dir'    : '../raw_data/geosse1_test',
    'pred_prefix' : 'sim.4'
}
my_all_args = my_all_args | my_prd_args
