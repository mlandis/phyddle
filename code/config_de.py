import scipy as sp

# helper variables
num_char = 3

#####################
# PIPELINE SETTINGS #
#####################
args = { 'job_name' : 'geosse_de_1' }

##################
# MODEL SETTINGS #
##################
mdl_args = {
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
args = args | mdl_args

######################
# SIMULATOR SETTINGS #
######################
sim_args = {
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
}
args = args | sim_args

#############################
# TENSOR-FORMATTER SETTINGS #
#############################
fmt_args = {
    'fmt_dir' : '../tensor_data',
    'tree_type'  : 'extant',
    'param_pred' : ['w_0', 'e_0', 'd_0_1', 'b_0_1', 'ed_0'],
    'param_data' : [],
    'tensor_format' : 'hdf5'
}
args = args | fmt_args

#####################
# LEARNING SETTINGS #
#####################
lrn_args = { 
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
args = args | lrn_args


#####################
# PLOTTING SETTINGS #
#####################
plt_args = {
    'plt_dir'        : '../plot',
    'network_prefix' : 'sim_batchsize128_numepoch20_nt200'
}
args = args | plt_args


#######################
# PREDICTING SETTINGS #
#######################

prd_args = {
    'pred_dir'    : '../raw_data/geosse1_test',
    'pred_prefix' : 'sim.4'
}
args = args | prd_args
