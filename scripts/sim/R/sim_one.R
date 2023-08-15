#!/usr/bin/env Rscript
library(ape)

# example command string
# cd ~/projects/phyddle/script
# ./sim/R/sim_one.R ../workspace/simulate/R_example/sim.0


# arguments
args        = commandArgs(trailingOnly = TRUE)
out_path    = args[1]
start_idx   = as.numeric(args[2])
batch_size  = as.numeric(args[3])
rep_idx     = start_idx:(start_idx+batch_size-1)
num_rep     = length(rep_idx)

# filesystem
tmp_fn  = paste0(out_path, "/sim.", rep_idx) # sim path prefix
phy_fn  = paste0(tmp_fn, ".tre")             # newick string
dat_fn  = paste0(tmp_fn, ".dat.nex")         # nexus string 
lbl_fn  = paste0(tmp_fn, ".param_row.csv")   # csv of params

print(tmp_fn)

#######################################
# You can do whatever you want below. #
#######################################

# birth-death model setup
birth           = runif(num_rep,0,1)
death           = birth * runif(num_rep)
max_time        = runif(num_rep,0,12)

# character model setup
num_char        = 2
num_states      = 3
state_rate      = runif(num_rep,0,1)
state_freqs     = rep(1/num_states, num_states)
state_labels    = 0:(num_states-1) #, collapse="")
root.value      = sample(x=num_states, size=num_rep,
                         prob=state_freqs, replace=T)

# simulate each replicate
for (i in 1:num_rep) {

    # simulate tree
    phy = rbdtree(birth=birth[i],
                  death=death[i],
                  Tmax=max_time[i])
    
    # simulate states for each character
    dat = c()
    for (j in 1:num_char) {
        dat_new = rTraitDisc(phy,
                             model="ER",
                             k=num_states,
                             rate=state_rate[i],
                             freq=state_freqs,
                             root.value=root.value[i],
                             states=state_labels)
        dat = cbind(dat, dat_new)
    }
    
    # ape::rTraitDisc ignores state_labels??
    # convert to base-0
    dat = dat - 1

    # construct training labels (paramaters)
    labels = c(birth[i], death[i], state_rate[i])
    names(labels) = c("birth", "death", "state_rate")
    df <- data.frame(t(labels))

    # save output
    write.tree(phy, file=phy_fn[i])
    write.nexus.data(dat, file=dat_fn[i], format="standard", datablock=TRUE)
    write.csv(df, file=lbl_fn[i], row.names=FALSE, quote=F)
}


###################################################
# Have phyddle provide useful messages concerning #
# files exist and formats are valid.              #
###################################################

# done!
quit()
