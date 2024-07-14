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
label_names = c( paste0("log_birth_",1:num_states), "log_death", "log_state_rate", "logit_sample_frac", "model_type", "start_state")

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
        sample_frac = 0.9999
        if (max_taxa > tree_width) {
            sample_frac = tree_width / max_taxa
        }
        logit_sample_frac = log( sample_frac/(1-sample_frac), base=10 )

        # simulate parameters
        model_type = sample(0:1, size=1)
        start_state = sample(1:2, size=1)
        log_state_rate = runif(1,-3,0)
        state_rate = 10^log_state_rate
        Q = matrix(state_rate,
                   ncol=num_states, nrow=num_states)
        diag(Q) = 0
        diag(Q) = -rowSums(Q)

        # Q = get_random_mk_transition_matrix(num_states, rate_model="ER", max_rate=0.1)
        log_birth = runif(n=num_states, -2, 0)
        if (model_type == 0) {
            log_birth[2] = log_birth[1]
        }
        birth = 10^log_birth

        death = min(birth) * 10^runif(n=1, -2, 0)
        death = rep(death, num_states)
        log_death = log(death[1], base=10)
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
    label_sim = c( log_birth[1], log_birth[2], log_death[1], log_state_rate, logit_sample_frac, model_type, start_state-1)
    names(label_sim) = label_names
    df_label = data.frame(t(label_sim))
    write.csv(df_label, file=lbl_fn[i], row.names=F, quote=F)

}


# done!
quit()
