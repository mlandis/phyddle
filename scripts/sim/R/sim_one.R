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
start_idx   = as.numeric(args[2])
batch_size  = as.numeric(args[3])
rep_idx     = start_idx:(start_idx+batch_size-1)
num_rep     = length(rep_idx)
get_mle     = FALSE

# filesystem
tmp_fn       = paste0(out_path, "/sim.", rep_idx) # sim path prefix
phy_fn       = paste0(tmp_fn, ".tre")             # newick file
dat_fn       = paste0(tmp_fn, ".dat.nex")         # csv of data
lbl_true_fn  = paste0(tmp_fn, ".param_row.csv")   # csv of true params
lbl_mle_fn   = paste0(tmp_fn, ".param_mle.csv")   # csv of estimated params

# dataset setup
num_states = 2
tree_width = 500
#param_names = c( paste0("birth_",1:num_states), paste0("death_",1:num_states), "state_rate" )
param_names = c( paste0("birth_",1:num_states), "death", "state_rate", "sample_frac" )

# simulate each replicate
for (i in 1:num_rep) {

    set.seed(rep_idx[i])

    num_taxa = 0
    while (num_taxa < 50) {
        
        # simulation conditions
        max_taxa = runif(1, 10, 5000)
        max_time = runif(1, 1, 100)
        sample_frac = 1.0
        if (max_taxa > tree_width) {
            cat(max_taxa,sample_frac,"\n")
            sample_frac = tree_width / max_taxa
            #max_taxa = tree_width
            cat(max_taxa,sample_frac,"\n")
        }

        # simulate parameters
        Q = get_random_mk_transition_matrix(num_states, rate_model="ER", max_rate=0.1)
        birth = runif(num_states, 0, 1)
        death = min(birth) * runif(1, 0, 1.0)
        death = rep(death, num_states)
        parameters = list(
            birth_rates=birth,
            death_rates=death,
            transition_matrix_A=Q
        )
        print(parameters)

        # simulate tree/data
        res_sim = simulate_dsse(
                Nstates=num_states,
                parameters=parameters,
                sampling_fractions=sample_frac,
                max_extant_tips=max_taxa,
                max_time=max_time,
                include_labels=T,
                no_full_extinction=T
        )

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

    # save data-generating params
    out_true = c(birth, death[1], Q[1,2], sample_frac)
    names(out_true) = param_names
    print(out_true)
    df_true = data.frame(t(out_true))
    write.csv(df_true, file=lbl_true_fn[i], row.names=F, quote=F)

    # MLE
    if (get_mle) {

        res_mle = NULL
        while (is.null(res_mle)) {
            first_guess = list(
                birth_rates = birth * exp(2 * runif(2, -0.5, 0.5)),
                death_rates = death * exp(2 * runif(2, -0.5, 0.5)),
                transition_matrix = exp(Q * runif(1, -0.5, 0.5))
            )
            print(res_sim$tip_states)

            res_mle = tryCatch(
            {
                #print("do-mle")
                # get MLE
                lower=list(
                    birth_rates=0,
                    death_rates=0,
                    transition_matrix=0
                )
                upper=list(
                    birth_rates=1,
                    death_rates=1,
                    transition_matrix=0.1
                )
                ret=fit_musse(
                    tree=tree_sim,
                    tip_pstates=state_sim+1,
                    transition_rate_model="ER",
                    first_guess=first_guess,
                    Nstates=num_states,
                    Ntrials=5,
                    Nscouts=10,
                    lower=lower,
                    upper=upper,
                    verbose=F,
                    diagnostics=F)

                #print("done")
                # return results
                ret$parameters
            },
            error = function(m) {
                NULL
            })
            rse_mle = 1

            # print res_mle
            #print(res_mle)
        }
        
        # save MLEs
        par_mle = res_mle
        names(par_mle)[ names(par_mle)=="transition_matrix" ] = "transition_matrix_A"
        out_mle = c( par_mle$birth_rates, par_mle$death_rates, par_mle$transition_matrix[1,2])
        names(out_mle) = param_names
        df_mle = data.frame(t(out_mle))
        write.csv(df_mle, file=lbl_mle_fn[i], row.names=FALSE, quote=F)
    }
}


# done!
quit()
