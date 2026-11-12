#install.packages("devtools")
#devtools::install_github("cmt2/RevGadgets")

library(RevGadgets)
library(ggplot2)

#####


file <- "empirical/sim.1.est.tre"

tree <- readTrees(path=file)
plot <- plotTree(tree = tree, line_width = 0.5)
labs <- c("0" = "Andean", 
          "1" = "Lowland", 
          "2" = "Both")

geo_exam <- processAncStates(file, state_labels = labs)
pie <- plotAncStatesPie(t =geo_exam, cladogenetic = TRUE, 
                        tip_labels_states_offset = .05,
                        # Offset the tip labels to make room for tip pies
                        tip_labels_offset = .2, 
                        # Move tip pies right slightly 
                        tip_pie_nudge_x = .07,
                        # Change the size of node and tip pies  
                        tip_pie_size = 0.5,
                        node_pie_size = .5, 
                        state_transparency = 1.0,
                        ladderize = FALSE) 
pdf("lio_ASR.pdf")
print(pie)
dev.off()


