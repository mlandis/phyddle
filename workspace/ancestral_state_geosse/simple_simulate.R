library(diversitree)
library(stringr)

# arguments
args        = commandArgs(trailingOnly = TRUE)
out_path    = args[1]
out_prefix  = args[2]
start_idx   = as.numeric(args[3])
batch_size  = as.numeric(args[4])
rep_idx     = start_idx:(start_idx+batch_size-1)

# filesystem
tmp_fn      = paste0(out_path, "/", out_prefix, ".", rep_idx)   # sim path prefix
phy_fn      = paste0(tmp_fn, ".tre")                            # newick file
dat_fn      = paste0(tmp_fn, ".dat.csv")                        # csv of data
lbl_fn      = paste0(tmp_fn, ".labels.csv")                     # csv of labels (e.g. params)
asr_true_fn = paste0(tmp_fn, ".anc_state.csv")                  # csv of ancestral states 

# Get the first state along a branch (forward time)
firstState <- function (mat) {
  mat[1,2]
}

# Get the last state along a branch (forward time)
lastState <- function(mat) {
  mat[dim(mat)[1], 2]
}

combine_states <- function(state_table) {
  paste0(state_table[1], state_table[2], state_table[3]) 
}

getHistory <- function(tree1, phy_file) {
  # Fix the off by one indexing for how the history is recorded 
  tree1$hist$from <- tree1$hist$from - 1
  tree1$hist$to   <- tree1$hist$to   - 1
  h <- history.from.sim.discrete(tree1, 0:2)
  
  # Find the first and last state for every branch
  mat_startState <- unlist(lapply(h$history, firstState))
  mat_endState   <- unlist(lapply(h$history, lastState))

  # Copy original tree
  newTree <- tree1
  
  # Create a matrix to hold the names of the children at each node
  parent_child_matrix <- matrix(0, nrow = tree$Nnode, ncol = 2,
                                dimnames = list(tree$node.label, c("child1", "child2")))
  
  # For every branch (edge) in the edge matrix
  # find the children labels
  for (i in 1:nrow(tree$edge)) {
    parent_node_num <- tree$edge[i, 1]
    child_node_num  <- tree$edge[i, 2]
    
    # Map node numbers to labels
    parent_label <- ifelse(parent_node_num <= length(tree$tip.label),
                           tree$tip.label[parent_node_num],
                           tree$node.label[parent_node_num - length(tree$tip.label)])
    child_label  <- ifelse(child_node_num <= length(tree$tip.label),
                           tree$tip.label[child_node_num],
                           tree$node.label[child_node_num - length(tree$tip.label)])
    
    # Determine if child is left or right child
    if (parent_child_matrix[parent_label, 1]  == 0) {
      parent_child_matrix[parent_label, 1] <- child_label
    } else {
      parent_child_matrix[parent_label, 2] <- child_label
      
    }
  }
  
  # Create a matrix to store the parent, children trios of states at internal nodes
  state_table <- matrix(NA, nrow = nrow(parent_child_matrix), ncol = 5)
  rownames(state_table) <- rownames(parent_child_matrix)
  colnames(state_table) <- c("parent", "child1", "child2", "child1_lb", "child2_lb")
  
  for(i in 1:nrow(parent_child_matrix)) {
    state_table[i,2] <- mat_startState[parent_child_matrix[i,1]]
    state_table[i,3] <- mat_startState[parent_child_matrix[i,2]]
    state_table[i,4] <- parent_child_matrix[i,1]
    state_table[i,5] <- parent_child_matrix[i,2]
      
    # If the node is not the root
    if (!is.na( mat_endState[rownames(parent_child_matrix)[i]])) {
      state_table[i,1] <- mat_endState[rownames(parent_child_matrix)[i]]  
      
    # If the node is the root
    } else {
      # If the two daughters are the same, the root is the
      # same state as the daughters
      if (state_table[i,2] == state_table[i,3]) {
        state_table[i,1] <- state_table[i,2]
        
        # Otherwise, the root is widespread
        # This is only for 2 regions
      } else {
        state_table[i,1] <- 0
      }
    }
    
  }

  num_tips <- Ntip(tree)
  num_nodes <- tree$Nnode
  
  # Select half of the nodes in the tree to rotate
  # This is needed for phyddle to have examples of both orderings
  nodes_to_rotate <- sample((num_tips + 1):(num_tips + num_nodes),
                            size = floor(num_nodes * 0.5),
                            replace = FALSE)
  
  for (node in nodes_to_rotate) {
    
    # Figure out which branches to rotate based on the parent index
    branchIndex <- which(newTree$edge[,1] == node)
    one <- branchIndex[1]
    two <- branchIndex[2]
    
    # Rotate branches
    tmp <-newTree$edge[one, ]
    newTree$edge[one, ] <- newTree$edge[two, ]
    newTree$edge[two, ] <- tmp
    
    # Rotate lengths
    tmp <- newTree$edge.length[one]
    newTree$edge.length[one] <- newTree$edge.length[two]
    newTree$edge.length[two] <- tmp
    
    # Update the table of left/right descendants 
    nodeName <- newTree$node.label[node-num_tips]
    tmp <- state_table[nodeName , 2]
    state_table[nodeName, 2] <- state_table[nodeName , 3]
    state_table[nodeName, 3] <- tmp
    
    tmp <- state_table[nodeName , 4]
    state_table[nodeName, 4] <- state_table[nodeName , 5]
    state_table[nodeName, 5] <- tmp
  }
  
  comb_state <- apply(state_table, 1, combine_states)
  newStates <- sapply(comb_state, switch, 
        "111" = "1", 
        "222" = "2", 
        "012" = "3",
        "021" = "4",
        "001" = "5",
        "010" = "6",
        "002" = "7",
        "020" = "8")

  sub_tree <- newTree
  write.tree(sub_tree, file = phy_file)

  return (cbind(newStates, state_table[,4:5]))
}

label_names = c("sA", "sB", "sAB", "xA", "xB", "dA", "dB")

nTrees <- start_idx - 1
nullTrees <- 0
noVarTree <- 0
set.seed(nTrees)
totTrees <- start_idx+batch_size-1
params <- matrix (0, nrow = totTrees, ncol = 7)
max_tax <- 52
i <- 1

while (nTrees < totTrees) {
  
  # Draw all parameters 
  sA  <- rexp(1, 1/.1)
  sB  <- rexp(1, 1/.1)
  xA  <- rexp(1, 1/.1)
  xB  <- rexp(1, 1/.1)
  dA  <- rexp(1, 1/.1)
  dB  <- rexp(1, 1/.1)
  sAB <- rexp(1, 1/.1)
  pars <- c(sA, sB, sAB, xA, xB, dA, dB)
  
  # Simulate tree
  tree <- tree.geosse(pars, max.taxa=max_tax, max.t=Inf, include.extinct=TRUE,
              x0=NA)
  # Find extant tips
  extant <-  which(str_detect(tree$tip.label, "sp"))

  # Check if tree exists
  if (is.null(tree) | sum(str_detect(tree$tip.label, "sp"))  != max_tax ) {
  	nullTrees <- nullTrees + 1
    
  # Confirm at least two tip states present
  } else if (length(table(tree$tip.state[extant])) < 3) {
    	noVarTree <- noVarTree + 1
    	
    	# Write all relevant files for phyddle
    } else {

    	stateHistory <- getHistory(tree, phy_fn[i])
    	# This file should actually be in the form
    	# node, state(1-8), node l, node r
    	write.table(stateHistory, file = asr_true_fn[i],quote = FALSE, col.names = FALSE, sep = "," )  
    	
    	df_state = data.frame(taxa=tree$tip.label, data=tree$tip.state, region1 = 0, region2 = 0)
	# In region one if state is either widespread or 1
    	df_state$region1[df_state$data == 0 | df_state$data == 1] <- 1
	# In region two if state is either widespread or 2
    	df_state$region2[df_state$data == 0 | df_state$data == 2] <- 1
    	
    	extant <- which(str_detect(df_state$taxa, "sp"))
    	df_state <- df_state[extant, ]
    	df_state <- df_state[ ,-2]

    	write.csv(df_state, file=dat_fn[i], row.names=F, quote=F)
    	  
    	# save learned labels (e.g. estimated data-generating parameters)
    	pars <- c(log(pars, base = 10))
    	names(pars) = label_names
    	df_label = data.frame(t(pars))
    	write.csv(df_label, file=lbl_fn[i], row.names=F, quote=F)
    	
    	nTrees <- nTrees + 1
    	
    	# Set the seed to get the same trees independent of how phyddle is run
    	set.seed(nTrees)
    	i <- i+1
    
  }
}

