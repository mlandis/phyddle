#!/usr/bin/env Rscript

library(castor)
library(ape)
library(rhdf5)
library(ggplot2)
library(reshape2)
library(cowplot)
library(parallel)

library(future)
library(future.apply)
source("validate_bisse_util.R")

# disable warnings
options(warn = -1)

# analysis settings
num_rep      = 100
num_cores    = availableCores() - 2

# directories
proj_name    = "bisse_validate_r"
sim_path     = paste0("../../workspace/", proj_name, "/simulate/")
fmt_path     = paste0("../../workspace/", proj_name, "/format/")
est_path     = paste0("../../workspace/", proj_name, "/estimate/")
#out_path     = paste0("../output/", proj_name, "/")
out_path     = paste0("./output/", proj_name, "/")

# files
param_true_fn = paste0(est_path, "out.test_true.labels_num.csv")
param_cnn_fn  = paste0(est_path, "out.test_est.labels_num.csv")
test_hdf5_fn  = paste0(fmt_path, "out.test.hdf5")

# true/test dataset
df_true = read.csv(param_true_fn, sep=",", header=T)
df_cnn  = read.csv(param_cnn_fn, sep=",", header=T)
df_true = df_true[ 1:num_rep, 2:ncol(df_true) ]
df_cnn_point = df_cnn[ 1:num_rep, seq(2,ncol(df_cnn),by=3) ]
df_cnn_lower = df_cnn[ 1:num_rep, seq(3, ncol(df_cnn), by=3) ]
df_cnn_upper = df_cnn[ 1:num_rep, seq(4, ncol(df_cnn), by=3) ]

# needed to clear dataset names for castor first_guess code
param_names       = names(df_true)
colnames(df_true) = NULL
colnames(df_cnn)  = NULL

# get test replicate indexes
test_idx            = h5read(file = test_hdf5_fn, name="idx", index=list(NULL))
test_aux_data       = h5read(file = test_hdf5_fn, name="aux_data", index=list(NULL,NULL))
test_aux_data_names = h5read(file = test_hdf5_fn, name="aux_data_names", index=list(NULL,NULL))
sample_frac_idx     = which(test_aux_data_names=="log_sample_frac")
sample_frac         = 10^test_aux_data[sample_frac_idx,]
rep_idx             = test_idx[1:num_rep]

# files for test replicates
tmp_fn = paste0(sim_path, "out.", rep_idx)  # sim path prefix
phy_fn = paste0(tmp_fn, ".tre")             # newick string
dat_fn = paste0(tmp_fn, ".dat.csv")         # nexus string
# ... can remove?
# phy_down_fn  = paste0(tmp_fn, ".downsampled.tre") # newick string
# mle_fn       = paste0("out_mle.csv")

my_input = list()
for (i in 1:length(rep_idx)) {
    my_input[[i]] = list(
        idx=rep_idx[i],
        phy_fn=phy_fn[i],
        sample_frac=sample_frac[i],
        dat_fn=dat_fn[i],
        par_true=unlist(as.vector(df_true[i,])),
        par_cnn_point=unlist(as.vector(df_cnn_point[i,])),
        par_cnn_lower=unlist(as.vector(df_cnn_lower[i,])),
        par_cnn_upper=unlist(as.vector(df_cnn_upper[i,])),
        param_names=param_names
    )
}

# MJL: uncomment to debug w/o parallel
# res_ind = list()
# for (i in 1:length(rep_idx)) {
#     res_ind[[i]] = bisse_mle( my_input[[i]] )    
# }
# bisse_mle( my_input[[100]] )    

# gather MLEs
future::plan("multisession", workers = num_cores)
res = future.apply::future_lapply(my_input, FUN = bisse_mle)

# format True/CNN/MLE parameter values
dat = make_input_table(res, unlog = T)

# save tables to file
save_tables(dat)

# gather test statistics
stat = make_compare_table(dat)

# plot results
plot_comparison(dat, stat)

# done!
