#!/usr/bin/env Rscript
library(castor)
library(ape)
library(dispRity)
library(extraDistr)

# disable warnings
options(warn = -1)

# example command string to simulate for "sim.1" through "sim.10"
# cd ~/projects/phyddle/scripts
# ./sim/R/sim_one.R ../workspace/simulate/R_example 1 10

# arguments
args        = commandArgs(trailingOnly = TRUE)
minTreeSize = as.numeric(args[1])
maxTreeSize = as.numeric(args[2])
out_path    = args[3]
out_prefix  = args[4]
start_idx   = as.numeric(args[5])
batch_size  = as.numeric(args[6])
rep_idx     = start_idx:(start_idx+batch_size-1)
num_rep     = length(rep_idx)
get_mle     = FALSE

# filesystem
tmp_fn      = paste0(out_path, "/", out_prefix, ".", rep_idx)   # sim path prefix
phy_fn      = paste0(tmp_fn, ".tre")                            # newick file
dat_fn      = paste0(tmp_fn, ".dat.csv")                        # csv of data
lbl_fn      = paste0(tmp_fn, ".labels.csv")                     # csv of labels (e.g. params)
asr_true_fn = paste0(tmp_fn, ".anc_state.csv")                  # csv of labels (e.g. params)

# dataset setup
num_states = 2
symm_Q_mtx = TRUE

# Set fixed tree size
tree_width = minTreeSize 
if (minTreeSize == maxTreeSize) {
	label_names = c(paste0("anc_state_", 1:(tree_width - 1)))
}

# simulate each replicate
for (i in 1:num_rep) {
 
	# set RNG seed
	set.seed(start_idx + i - 1)
	
	# rejection sample
	num_taxa = 0
	while (num_taxa < tree_width +1) {

		# Draw tree size
		if (minTreeSize != maxTreeSize) {
			tree_width <- rdunif(1, minTreeSize, maxTreeSize)
			label_names = c(paste0("anc_state_", 1:(tree_width - 1)))
		}
		
		log_birth = runif(1, -2, 0)
		birth = 10^log_birth
		
		death = min(birth) * 10^runif(n=1, -2, 0)
		log_death = log(death[1], base=10)
		
		
		res_sim = generate_tree_hbds(max_extant_tips = tree_width + 1, 
		                            include_extant = TRUE, 
		                            lambda = birth, 
		                            mu = death 
		                  )
		
		# check if tree is valid
		num_taxa = length(res_sim$tree$tip.label)
	}

    
	# save tree
	tree <- res_sim$tree

	# drop one of tips with zero branch length
	edge <- (which(tree$edge.length == 0)[1])
	drop <- (tree$edge[edge,2 ])
	tree_sim <- drop.tip(res_sim$tree, drop, trim.internal = TRUE, colapse.singles = TRUE  )

	tree_sim <- makeNodeLabel(tree_sim, method = "number", prefix = "node")
	write.tree(tree_sim, file=phy_fn[i])

	# transition simulate parameters
	log_state_rate = runif(1,-3,0)
	state_rate = 10^log_state_rate
	Q = matrix(state_rate,
	           ncol=num_states, nrow=num_states)
	diag(Q) = 0
	diag(Q) = -rowSums(Q)
	characterData <- simulate_mk_model(tree_sim, Q, root_probabilities="stationary",
	              include_tips=TRUE, include_nodes=TRUE,
	              Nsimulations=1, drop_dims=TRUE)
	
	# save data
	state_sim = characterData$tip_states - 1
	state_sim_node = characterData$node_states - 1
	
	df_state = data.frame(taxa=tree_sim$tip.label, data=state_sim)
	write.csv(df_state, file=dat_fn[i], row.names=F, quote=F)

	# save learned labels (e.g. estimated data-generating parameters)
	label_sim =  state_sim_node
	names(label_sim) = label_names
	df_label = data.frame(t(label_sim))

	write.csv(df_label, file=lbl_fn[i], row.names=F, quote=F)
	trueRename <- df_label
	colnames(trueRename) <- paste("node", c(1:(tree_width -1)), sep = "")
	write.table(t(trueRename), file = asr_true_fn[i], quote =F, col.names =F, sep = ',')
	
}


# done!
