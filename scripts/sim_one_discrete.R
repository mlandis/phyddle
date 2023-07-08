library(ape)

# arguments
args   = commandArgs(trailingOnly = TRUE)
tmp_fn = args[1]
#tmp_fn = "/Users/mlandis/projects/phyddle/workspace/raw_data/n2s3_discrete_R/sim.0"

# setup
num_char        = 2
num_states      = 3
num_state_pairs = num_states*(num_states-1)
birth           = runif(1,0,1)
death           = birth * runif(1)
max_time        = runif(1,0,12)
state_rate      = runif(1,0,1)
if (F) {
    state_rates     = runif(num_state_pairs/2, 0, 1)
    q_labels        = rep("", num_state_pairs/2)
    state_Q_mtx     = matrix(0, nrow=num_states, ncol=num_states)
    k = 1
    for (i in 1:num_states) {
        for (j in 1:num_states) {
            if (j > i) {
                state_Q_mtx[i,j] = state_rates[k]
                state_Q_mtx[j,i] = state_Q_mtx[i,j]
                q_labels[k] = paste0("q_",j,"_",i)
                k = k + 1
            }
        }
    }
    diag(state_Q_mtx) = 0
    diag(state_Q_mtx) = -rowSums(state_Q_mtx)
}
state_freqs     = rep(1/num_states, num_states) # runif(num_states, 0, 1)
state_freqs     = state_freqs / sum(state_freqs)
state_labels    = 0:(num_states-1) #, collapse="")
root.value      = sample(x=num_states, size=1, prob=state_freqs)

# filesystem
phy_fn  = paste0(tmp_fn, ".tre")
dat_fn  = paste0(tmp_fn, ".dat.nex")
lbl_fn  = paste0(tmp_fn, ".param_row.csv")

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

# construct labels
#k = 1
#for (i in 1:num_states) {
#    for (j in 1:num_states) {
#        if (j > i) {
#        }
#    }
#}
#pi_labels = paste0("pi_", 1:num_states)

labels = c(birth, death, state_rate)
names(labels) = c("birth", "death", "state_rate")
df <- data.frame(t(labels))


# save output
write.tree(phy, file=phy_fn)
write.nexus.data(dat, file=dat_fn, format="standard", datablock=TRUE)
write.csv(df, file=lbl_fn, row.names=FALSE, quote=F)


# done!
quit()
