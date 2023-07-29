library(ape)

# This script assumes that the first trailing argument will be the
# prefix for the simulated replicate. For example, if we called
#
#     Rscript simulate/sim_cont_one.R ../workspace/simulate/my_project/sim.1
#
# where the simulation prefix is stored in the R variable, tmp_fn. For that
# value of tmp_fn, the script will generate the following files
#
#     ../workspace/simulate/my_project/sim.1.tre
#     ../workspace/simulate/my_project/sim.1.dat.csv
#     ../workspace/simulate/my_project/sim.1.labels.csv
#

# arguments
args   = commandArgs(trailingOnly = TRUE)
tmp_fn = args[1]

# setup
num_char        = c(2,3)
num_states      = c(2,4)
num_state_pairs = num_states*(num_states-1)
birth           = runif(1,0,1)
death           = birth * runif(1)
max_time        = runif(1,0,12)
state_rate      = runif(1,0,1)
trait_rate      = runif(1,0,1)
alpha           = rexp(1)
theta           = rnorm(1)
sigma           = rexp(1, 0.1)

disc_root_value = c()
state_freqs = list()
for (i in 1:num_char[1]) {
    state_freqs[[i]] = rep(1/num_states[i], num_states[i])
    state_labels = 0:(num_states[i]-1)
    names(state_freqs[[i]]) = state_labels
    disc_root_value[i] = sample(x=num_states[i], size=1, prob=state_freqs[[i]])
}
cont_root_value = c()
for (i in 1:num_char[2]) {
    cont_root_value[i] = rnorm(1)
}

# filesystem
phy_fn  = paste0(tmp_fn, ".tre")
dat_fn  = paste0(tmp_fn, ".dat.csv")
lbl_fn  = paste0(tmp_fn, ".param_row.csv")

# simulate tree
phy = rbdtree(birth=birth, death=death, Tmax=max_time)

# simulate discrete states
dat = c()
for (i in 1:num_char[1]) {
    dat_new = rTraitDisc(phy,
                         model="ER",
                         k=num_states[i],
                         rate=state_rate,
                         freq=state_freqs[[i]],
                         states=names(state_freqs[[i]]),
                         root.value=disc_root_value[i])
    dat = cbind(dat, dat_new)
}
# convert to base-0 because ape::rTraitDisc ignores state_labels??
dat = dat - 1

# simulate continuous traits
for (i in 1:num_char[2]) {
    dat_new = rTraitCont(phy,
                         model="OU",
                         sigma=sigma,
                         alpha=alpha,
                         theta=theta,
                         root.value=cont_root_value[i])
    dat = cbind(dat, dat_new)
}


# construct training labels
labels = c(birth, death, state_rate, sigma, theta, alpha)
names(labels) = c("birth", "death", "state_rate", "sigma", "theta", "alpha") # "root_value")
names(disc_root_value) = paste0("disc_root_value", 1:num_char[1])
labels = labels(disc_root_value)
names(cont_root_value) = paste0("cont_root_value", 1:num_char[2])
labels = labels(cont_root_value)
df <- data.frame(t(labels))


# save output
write.tree(phy, file=phy_fn)
write.csv(dat, file=dat_fn, row.names=F, quote=F)
write.csv(df, file=lbl_fn, row.names=F, quote=F)


# done!
quit()
