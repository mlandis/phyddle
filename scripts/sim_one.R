library(ape)

# arguments
args   = commandArgs(trailingOnly = TRUE)
tmp_fn = args[1]

# setup
num_states      = 3
num_state_pairs = num_states*(num_states-1)
birth           = runif(1,0,1)
death           = birth * runif(1)
max_time        = runif(1,0,10)
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
state_freqs     = rep(1/num_states, num_states) # runif(num_states, 0, 1)
state_freqs     = state_freqs / sum(state_freqs)
state_labels    = paste0(1:num_states, collapse="")
root.value      = sample(x=num_states, size=1, prob=state_freqs)

# filesystem
phy_fn  = paste0(tmp_fn, ".tre")
dat_fn  = paste0(tmp_fn, ".dat.nex")
lbl_fn  = paste0(tmp_fn, ".param_row.csv")

# simulate tree
phy = rbdtree(birth=birth, death=death, Tmax=max_time)

# simulate data
dat = rTraitDisc(phy,
                 model="SYM",
                 k=num_states,
                 rate=state_rates,
                 freq=state_freqs,
                 states=1:state_labels)

# construct labels
k = 1
for (i in 1:num_states) {
    for (j in 1:num_states) {
        if (j > i) {
        }
    }
}
pi_labels = paste0("pi_", 1:num_states)

labels = c(birth, death, state_rates)
names(labels) = c("birth", "death", q_labels)
df <- data.frame(t(labels))


# save output
write.tree(phy, file=phy_fn)
write.nexus.data(dat, file=dat_fn, format="standard", datablock=TRUE)
write.csv(df, file=lbl_fn, row.names=FALSE, quote=F)


# done!
quit()
