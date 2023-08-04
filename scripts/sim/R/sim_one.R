#!/usr/bin/env Rscript
library(ape)

# example command string
# cd ~/projects/phyddle/script
# ./sim/R/sim_one.R ../workspace/simulate/R_example/sim.0

# arguments
args   = commandArgs(trailingOnly = TRUE)
tmp_fn = args[1]

# filesystem
phy_fn  = paste0(tmp_fn, ".tre") # newick string
dat_fn  = paste0(tmp_fn, ".dat.nex") # nexus string (will support csv)
lbl_fn  = paste0(tmp_fn, ".param_row.csv") # csv of parameter values

#######################################
# You can do whatever you want below. #
#######################################

# birth-death model setup
birth           = runif(1,0,1)
death           = birth * runif(1)
max_time        = runif(1,0,12)

# character model setup
num_char        = 2
num_states      = 3
state_rate      = runif(1,0,1)
state_freqs     = rep(1/num_states, num_states)
state_labels    = 0:(num_states-1) #, collapse="")
root.value      = sample(x=num_states, size=1, prob=state_freqs) # how to pass in

# simulate tree
phy = rbdtree(birth=birth, death=death, Tmax=max_time)

# simulate data
dat = c()
for (i in 1:num_char) {
    dat_new = rTraitDisc(phy,
                         model="ER",
                         k=num_states,
                         rate=state_rate,
                         freq=state_freqs,
                         states=state_labels)
    dat = cbind(dat, dat_new)
}
# convert to base-0 because ape::rTraitDisc ignores state_labels??
dat = dat - 1

# construct training labels (paramaters)
labels = c(birth, death, state_rate)
names(labels) = c("birth", "death", "state_rate")
df <- data.frame(t(labels))

# save output
write.tree(phy, file=phy_fn)
write.nexus.data(dat, file=dat_fn, format="standard", datablock=TRUE)
write.csv(df, file=lbl_fn, row.names=FALSE, quote=F)

###################################################
# Have phyddle provide useful messages concerning #
# files exist and formats are valid.              #
###################################################

# done!
quit()
