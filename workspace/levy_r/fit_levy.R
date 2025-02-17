#!/usr/bin/env Rscript
#library(neldermead)
library(devtools)
devtools::install_local("~/projects/pulsR/")
library(pulsR)
#source("levy_pruning_optim.r")
#source("levy_pruning_cf.r")
#source("levy_pruning_prob.r")
#source("levy_pruning_tools.r")
library(statmod)
library(castor)
library(ape)
library(geiger)
library(data.table)

# disable warnings
options(warn = -1)

# args
args = commandArgs(trailingOnly=T)
print(args)
i = as.numeric(args[1])
mdl_est = args[2]

# read in dataset
num_attempt = 4
test_true   = read.csv("./estimate/out.test_true.labels_cat.csv", header=T, sep=",")
idx         = test_true$idx[ which(i == test_true$idx) ]
mdl_true    = test_true$model_type[ which(i==test_true$idx) ]
tmp_fn      = "./simulate/out."

phy_fn = paste0(tmp_fn, idx, ".tre")               # newick file
dat_fn = paste0(tmp_fn, idx, ".dat.csv")           # csv of data
print(phy_fn)
print(dat_fn)
phy = read.tree(phy_fn)
df = read.csv(dat_fn)
dat = df[,2]
names(dat) = df[,1]

fit<-fit_reml_levy(phy, dat, model = mdl_est, silent=F, num_attempt = num_attempt) #, maxfun=10, maxiter=10)

mdl_names = c("BM","OU","EB","NIG")
#if (mdl_est %in% c("JN","VG","NIG","BMJN","BMVG","BMNIG")) {
#    mdl_est = "Levy"
#}
mdl_est_idx = which(mdl_est == mdl_names) - 1
print(idx)
print(mdl_true)
print(mdl_est_idx)
print(fit$AIC)
print(fit$params)
#vec = c(idx, mdl_true, mdl_est_idx, fit$AIC, fit$params)
res = data.frame( matrix(c(idx, mdl_true, mdl_est_idx, fit$AIC, unlist(fit$params)), nrow=1) )
print(ncol(res))
colnames(res) = c("idx", "model_true", "model_est", "aic", names(fit$params))
out_fn = paste0("./pulsr/result.", mdl_est, ".",  idx, ".csv")
write.csv(res, out_fn, quote=F, row.names=F)

quit()

