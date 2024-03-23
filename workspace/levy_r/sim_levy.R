#!/usr/bin/env Rscript
library(pulsR)
library(statmod)
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
                "log10_sigma_tip",
                "frac_of_var",
                "sample_frac",
                "trait_orig",
                "trait_min",
                "trait_max",
                "model_type")

# simulate each replicate
for (i in 1:num_rep) {

    # set RNG seed
    set.seed(rep_idx[i])

    # rejection sample
    num_taxa = 0
    min_taxa = 10
    while (num_taxa < min_taxa) {
        
        # simulation conditions
        max_taxa = runif(1, 50, 1000)
        max_time = runif(1, 0.1, 2)
        sample_frac = 1.0
        if (max_taxa > tree_width) {
            sample_frac = tree_width / max_taxa
        }

        # model type
        model_names = c("BM","OU","EB","JN","NIG","BMJN","BMNIG")
        num_models = length(model_names)
        model_type = sample(0:(num_models-1), size=1)
        model_type = 3

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
        num_taxa = length(phy$tip.label)
        if (num_taxa < min_taxa) {
            next
        }

        # Levy parameters
        trait_orig = rnorm(1, 0, 2)
        process_var = rexp(1, 1)
        sigma_tip = rexp(1, 5)
        halflife = rgamma(1, 15, 2)
        decay = -rgamma(1, 15, 2)
        if (model_type == 0) {
            # BM-only
            model = "BM"
            frac_of_var = rbeta(1, 1, 10000)  # all BM
            process_kurt = rexp(1, 10000)     # ~0 kurtosis
        } else if (model_type == 1) {
            # OU
            model = "OU"
            frac_of_var = rbeta(1, 1, 10000)  # all OU
            process_kurt = rexp(1, 10000)     # ~0 kurtosis
        } else if (model_type == 2) {
            # EB
            model = "EB"
            frac_of_var = rbeta(1, 1, 10000)  # all EB
            process_kurt = rexp(1, 10000)       # ~0 kurtosis
        } else if (model_type == 3) {
            # JN-only
            model = "JN"
            frac_of_var = rbeta(1, 10000, 1)  # all JN
            process_kurt = rgamma(1, 4, 0.4)    # ~8 kurtosis
        } else if (model_type == 4) {
            # NIG-only
            model = "NIG"
            frac_of_var = rbeta(1, 10000, 1)  # all NIG
            process_kurt = rgamma(1, 4, 0.4)  # ~8 kurtosis
        } else if (model_type == 5) {
            # BM+JN
            model = "BMJN"
            frac_of_var = rbeta(1, 6, 2)      # 25/75 BM/JN
            process_kurt = rgamma(1, 4, 0.4)  # ~8 kurtosis
        } else if (model_type == 6) {
            # BM+NIG
            model = "BMNIG"
            frac_of_var = rbeta(1, 6, 2)      # 25/75 BM/NIG
            process_kurt = rgamma(1, 4, 0.4)  # ~8 kurtosis
        }
        params = pulsR::get_params_for_var(process_var=process_var,
                                           process_kurt=process_kurt,
                                           frac_of_var=frac_of_var,
                                           halflife=halflife,
                                           decay=decay)

        par = list()
        for (p in names(params[[tolower(model)]])) {
            v = params[[tolower(model)]][[p]]
            p_new = gsub("\\.","_",p)
            par[[p_new]] = v
        }
        par$sigma_tip = sigma_tip
        if (model == "OU") {
            par$theta_ou = trait_orig
        }

        # simulate traits
        traits = rlevy(model=model, n=1, phy=phy, par=par)
        traits = traits + trait_orig
        trait_min = min(traits)
        trait_max = max(traits)
        if (trait_min != trait_max) {
            traits = (traits - trait_min) / (trait_max - trait_min)
        } else {
            traits = rep(0, length(traits))
        }

    }
   
    # save tree
    tree_sim = phy
    write.tree(tree_sim, file=phy_fn[i])

    # save data
    state_sim = traits
    df_state = data.frame(taxa=tree_sim$tip.label, data=state_sim)
    write.csv(df_state, file=dat_fn[i], row.names=F, quote=F)

    # save learned labels (e.g. estimated data-generating parameters)
    label_sim = c(process_var, process_kurt, sigma_tip,
                  frac_of_var, sample_frac,
                  trait_orig, trait_min, trait_max,
                  model_type)
    label_sim[1:3] = log(label_sim[1:3], base=10)
    names(label_sim) = label_names
    df_label = data.frame(t(label_sim))
    write.csv(df_label, file=lbl_fn[i], row.names=F, quote=F)

}


# done!
quit()
