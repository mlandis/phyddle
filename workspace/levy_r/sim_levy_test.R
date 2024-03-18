#!/usr/bin/env Rscript
library(pulsR)
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
tree_width = 500
label_names = c("log10_process_var",
                "log10_process_kurt",
                "log10_frac_of_var",
                "log10_sigma_bm",
                "log10_lambda_jn",
                "log10_delta_jn",
                "log10_sample_frac",
                "trait_orig",
                "trait_min",
                "trait_max")

# simulate each replicate
for (i in 1:num_rep) {

    # set RNG seed
    set.seed(rep_idx[i])

    # rejection sample
    num_taxa = 0
    while (num_taxa < 10) {
        
        # simulation conditions
        max_taxa = runif(1, 50, 1000)
        max_time = runif(1, 0, 1)
        sample_frac = 1.0
        if (max_taxa > tree_width) {
            sample_frac = tree_width / max_taxa
        }

        # birth-death parameters
        birth = runif(1, 1, 5)
        death = min(birth) * runif(1, 0, 1)

        # simulate tree
		res = generate_tree_hbd_reverse(Ntips=max_taxa,
                                        crown_age=max_time,
                                        lambda=birth,
                                        mu=death,
                                        rho=1.0)
        phy = res$trees[[1]]

        # Levy parameters
        trait_orig = rnorm(1, 0, 2)
        process_var = rexp(1, 1)
        process_kurt = rexp(1, 0.1)
        frac_of_var = runif(1, 0, 1)
        frac_of_var = 1
        params = pulsR::get_params_for_var(process_var=process_var,
                                           process_kurt=process_kurt,
                                           frac_of_var=frac_of_var)
        sigma_bm = params$bmjn$sigma.bm
        lambda_jn = params$bmjn$lambda.jn
        delta_jn = params$bmjn$delta.jn


        # simulate traits
        traits = rlevy(model="BMJN", n=1, phy=phy, par=list(sigma_bm=sigma_bm, lambda_jn=lambda_jn, delta_jn=delta_jn))
        traits = traits + trait_orig
        trait_min = min(traits)
        trait_max = max(traits)
        if (trait_min != trait_max) {
            traits = (traits - trait_min) / (trait_max - trait_min)
        } else {
            traits = rep(0, length(traits))
        }

        # check if tree is valid
        num_taxa = length(phy$tip.label)
    }
   
    # save tree
    tree_sim = phy
    write.tree(tree_sim, file=phy_fn[i])

    # save data
    state_sim = traits
    df_state = data.frame(taxa=tree_sim$tip.label, data=state_sim)
    write.csv(df_state, file=dat_fn[i], row.names=F, quote=F)

    # save learned labels (e.g. estimated data-generating parameters)
    label_sim = c(process_var, process_kurt, frac_of_var,
                  sigma_bm, lambda_jn, delta_jn,
                  sample_frac,
                  trait_orig, trait_min, trait_max)
    label_sim[1:7] = log(label_sim[1:7], base=10)
    names(label_sim) = label_names
    df_label = data.frame(t(label_sim))
    write.csv(df_label, file=lbl_fn[i], row.names=F, quote=F)

}


# done!
quit()
