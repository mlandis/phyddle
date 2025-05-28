#!/usr/bin/env Rscript
#library(phytools)
library(castor)
library(ape)
library(dispRity)

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

set.seed(start_idx)

# filesystem
tmp_fn = paste0(out_path, "/", out_prefix, ".", rep_idx)   # sim path prefix
phy_fn = paste0(tmp_fn, ".tre")               # newick file
dat_fn = paste0(tmp_fn, ".dat.csv")           # csv of data
lbl_fn = paste0(tmp_fn, ".labels.csv")        # csv of labels (e.g. params)
asr_true_fn = paste0(tmp_fn, ".anc_state.csv")     # csv of labels (e.g. params)
asr_fn = paste0(tmp_fn, ".asr.csv")        # csv of labels (e.g. params)
age_fn = paste0(tmp_fn, ".age.csv")        # csv of node ages 
age_prop_fn = paste0(tmp_fn, ".age_prop.csv")        # csv of node ages 

# dataset setup
num_states = 2
symm_Q_mtx = TRUE
tree_min_width= 10
tree_max_width=100
#tree_width = 500

# simulate each replicate
# print(num_rep)
for (i in 1:num_rep) {
 
     # set RNG seed
    set.seed(rep_idx + i - 1)
     # rejection sample
     num_taxa = 0
	tree_width <- floor(runif(1, tree_min_width, tree_max_width + 1))

	label_names = c(paste0("anc_state_", 1:(tree_width - 1)))
     while (num_taxa < tree_width +1) {
         
         # simulation conditions
         max_taxa = tree_width + 1
         #max_taxa = runif(1, 10, 5000)
         max_time = runif(1, 1, 1000)
         #sample_frac = 0.9999
         #if (max_taxa > tree_width) {
         #    sample_frac = tree_width / max_taxa
         #}
         #logit_sample_frac = log( sample_frac/(1-sample_frac), base=10 )
 
         # simulate parameters
         #model_type = sample(0:1, size=1)
         start_state = sample(1:2, size=1)
         log_state_rate = runif(1,-3,0)
         state_rate = 10^log_state_rate
       # This has the forward and backwards rates equal, which is debatable whether we 
       # want that
         Q = matrix(state_rate,
                    ncol=num_states, nrow=num_states)
         diag(Q) = 0
         diag(Q) = -rowSums(Q)
#	 print(Q)
 
         #Q = get_random_mk_transition_matrix(num_states, rate_model="ER", max_rate=0.1)
         log_birth = runif(1, -2, 0)
         birth = 10^log_birth
 
         death = min(birth) * 10^runif(n=1, -2, 0)
         log_death = log(death[1], base=10)
 
 
       res_sim = generate_tree_hbds(max_extant_tips = max_taxa, 
                                    max_time = max_time, 
                                    include_extant = TRUE, 
                                    lambda = birth, 
                                    mu = death 
                          )
       # You probably want to include a sampling fractions
 
 
         # check if tree is valid
         num_taxa = length(res_sim$tree$tip.label)
     }
    
     # save tree
 #     # save tree
#     print(res_sim$tree)
     tree <- res_sim$tree
     edge <- (which(tree$edge.length == 0)[1])
     drop <- (tree$edge[edge,2 ])
     tree_sim <- drop.tip(res_sim$tree, drop, trim.internal = TRUE, colaspe.singles = TRUEpe  )
 


    tree_sim <- makeNodeLabel(tree_sim, method = "number", prefix = "node")
    write.tree(tree_sim, file=phy_fn[i])
            Q = matrix(0.2,
                       ncol=2, nrow=2)
            diag(Q) = 0
            diag(Q) = -rowSums(Q)

    characterData <- simulate_mk_model(tree_sim, Q, root_probabilities="stationary",
                  include_tips=TRUE, include_nodes=TRUE,
                  Nsimulations=1, drop_dims=TRUE)

    # save data
    state_sim = characterData$tip_states - 1
    state_sim_node = characterData$node_states - 1
    #state_sim = res_sim$tip_states - 1

    df_state = data.frame(taxa=tree_sim$tip.label, data=state_sim)
    write.csv(df_state, file=dat_fn[i], row.names=F, quote=F)
    asr_est <- asr_mk_model(tree_sim, characterData$tip_states, Nstates = 2, rate_model = "ARD" ) 

    # save learned labels (e.g. estimated data-generating parameters)
    label_sim =  state_sim_node
    names(label_sim) = label_names
    df_label = data.frame(t(label_sim))
    asr_est_print <- data.frame((asr_est$ancestral_states - 1))
    rownames(asr_est_print) <- tree_sim$node.label

    write.table(asr_est_print, file = asr_fn[i], sep = ',', col.names =F, row.names =T, quote = F)
    write.csv(df_label, file=lbl_fn[i], row.names=F, quote=F)
    trueRename <- df_label
    colnames(trueRename) <- paste("node", c(1:(tree_width -1)), sep = "")
    write.table(t(trueRename), file = asr_true_fn[i], quote =F, col.names =F, sep = ',')

    ages <- (tree.age(tree_sim))
    ages <- ages[(dim(asr_est_print)[1]+2):dim(ages)[1],]
    write.csv(ages, file = age_fn[i], quote= F, row.names =F)
    agesProp <- ages[,1] / max(ages[,1])
    names(agesProp) <- ages[,2]
    agesProp <- t(agesProp)
    write.csv(agesProp, file = age_prop_fn[i], quote= F, row.names =F)

}


# done!
