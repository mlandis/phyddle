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
num_rep      = 250
num_cores    = availableCores() - 2

# directories
proj_name    = "example"
sim_path     = paste0("../../workspace/simulate/", proj_name, "/")
fmt_path     = paste0("../../workspace/format/", proj_name, "/")
est_path     = paste0("../../workspace/estimate/", proj_name, "/")
out_path     = paste0("../output/", proj_name, "/")

# files
param_true_fn = paste0(est_path, "new.1.test_true.labels.csv")
param_cnn_fn  = paste0(est_path, "new.1.test_est.labels.csv")
test_hdf5_fn  = paste0(fmt_path, "test.nt500.hdf5")

# true/test dataset
df_true = read.csv(param_true_fn, sep=",", header=T)
df_cnn  = read.csv(param_cnn_fn, sep=",", header=T)
df_true = df_true[ 1:num_rep, ]
df_cnn  = df_cnn[ 1:num_rep, seq(1,ncol(df_cnn),by=3) ]

# needed to clear dataset names for castor first_guess code
param_names       = names(df_true)
colnames(df_true) = NULL
colnames(df_cnn)  = NULL

# get test replicate indexes
test_idx            = h5read(file = test_hdf5_fn, name="idx", index=list(NULL))
test_aux_data       = h5read(file = test_hdf5_fn, name="aux_data", index=list(NULL,NULL))
test_aux_data_names = h5read(file = test_hdf5_fn, name="aux_data_names", index=list(NULL,NULL))
sample_frac_idx     = which(test_aux_data_names=="sample_frac")
sample_frac         = test_aux_data[sample_frac_idx,]
rep_idx             = test_idx[1:num_rep]

# files for test replicates
tmp_fn = paste0(sim_path, "sim.", rep_idx)  # sim path prefix
phy_fn = paste0(tmp_fn, ".tre")             # newick string
dat_fn = paste0(tmp_fn, ".dat.nex")         # nexus string
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
        par_cnn=unlist(as.vector(df_cnn[i,])),
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
dat = make_input_table(res)

# save tables to file
save_tables(dat)

# gather test statistics
stat = make_compare_table(dat)

# plot results
plot_comparison(dat, stat)

# done!
