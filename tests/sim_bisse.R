#!/usr/bin/env Rscript
#library(phytools)
library(castor)
library(ape)

# disable warnings
options(warn = -1)

# example command string to simulate for "sim.1" through "sim.10"
# cd ~/projects/phyddle/scripts
# ./sim/R/sim_one.R ../workspace/simulate/R_example 1 10

# arguments
args        = commandArgs(trailingOnly = TRUE)
out_path    = args[1]
out_prefix  = args[2]
start_idx   = as.numeric(args[3])
batch_size  = as.numeric(args[4])
rep_idx     = start_idx:(start_idx+batch_size-1)
num_rep     = length(rep_idx)
get_mle     = FALSE

# filesystem
tmp_fn = paste0(out_path, "/", out_prefix, ".", rep_idx)   # sim path prefix
phy_fn = paste0(tmp_fn, ".tre")               # newick file
dat_fn = paste0(tmp_fn, ".dat.csv")           # csv of data
lbl_fn = paste0(tmp_fn, ".labels.csv")        # csv of labels (e.g. params)

# dataset setup
num_states = 2
symm_Q_mtx = TRUE
tree_width = 500
label_names = c( paste0("log10_birth_",1:num_states), "log10_death", "log10_state_rate", "log10_sample_frac", "model_type", "start_state")

# simulate each replicate
for (i in 1:num_rep) {

    # set RNG seed
    set.seed(rep_idx[i])

    # rejection sample
    num_taxa = 0
    while (num_taxa < 10) {
        
        # simulation conditions
        max_taxa = runif(1, 10, 5000)
        max_time = runif(1, 1, 100)
        sample_frac = 1.0
        if (max_taxa > tree_width) {
            sample_frac = tree_width / max_taxa
        }

        # simulate parameters
        model_type = sample(0:1, size=1)
        start_state = sample(1:2, size=1)
        Q = matrix(runif(n=num_states*num_states, 0, 0.1),
                   ncol=num_states, nrow=num_states)
        diag(Q) = 0
        if (symm_Q_mtx) {
            Q[lower.tri(Q)] = t(Q)[lower.tri(Q)]
        }
        diag(Q) = -rowSums(Q)

        # Q = get_random_mk_transition_matrix(num_states, rate_model="ER", max_rate=0.1)
        birth = runif(n=num_states, 0, 1 )  # rate=1.0)
        if (model_type == 0) {
            birth[2] = birth[1]
        }
        death = min(birth) * runif(n=1, 0, 1)  # rate=1.0)
        death = rep(death, num_states)
        parameters = list(
            birth_rates=birth,
            death_rates=death,
            transition_matrix_A=Q
        )

        # simulate tree/data
        res_sim = simulate_dsse(
                Nstates=num_states,
                parameters=parameters,
                start_state=start_state,
                sampling_fractions=sample_frac,
                max_extant_tips=max_taxa,
                max_time=max_time,
                include_labels=T,
                no_full_extinction=T)

        # check if tree is valid
        num_taxa = length(res_sim$tree$tip.label)
    }
   
    # save tree
    tree_sim = res_sim$tree
    write.tree(tree_sim, file=phy_fn[i])

    # save data
    state_sim = res_sim$tip_states - 1
    df_state = data.frame(taxa=tree_sim$tip.label, data=state_sim)
    write.csv(df_state, file=dat_fn[i], row.names=F, quote=F)

    # save learned labels (e.g. estimated data-generating parameters)
    label_sim = c( birth[1], birth[2], death[1], Q[1,2], sample_frac, model_type, start_state-1)
    label_sim[1:5] = log(label_sim[1:5], base=10)
    # label_sim[5] = label_sim[5] / (1 - exp(label_sim[5]))
    names(label_sim) = label_names
    df_label = data.frame(t(label_sim))
    write.csv(df_label, file=lbl_fn[i], row.names=F, quote=F)

}


# done!
quit()
