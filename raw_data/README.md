# Raw data

The `raw_data` directory contains directories of simulated training replicates for each model. Subdirectories represent different pipeline projects.

The Simulating step will create the following files for simulation replicate `i`:
```
sim.i.tre             # Newick phylogeny string
sim.i.dat.nex         # Nexus character matrix
sim.i.param_row.csv   # simulation parameters with one parameter per row
sim.i.param_col.csv   # simulation parameters with one parameter per column
```

The Formatting step additionally creates the following files when preparing the encoded tensor:
```
sim.i.info.csv        # information about replicate settings
sim.i.extant.tre      # Newick phylogeny string with only extant taxa
                      # when 'tree_type' == 'extant'
```

Phylogenetic-state tensor encodings are also stored only when `'save_phyenc_csv' == True`:
```
sim.i.cdvs.tre        # compact diversity vector + brlen info expansion + states
                      # when 'tree_type' == 'extant'
sim.i.cblvs.tre       # compact bijective ladderized vector + brlen info expansion + states
                      # when 'tree_type' == 'serial')
```

When MASTER is used to simulate data, the follow additional files may appear:
```
sim.i.xml             # MASTER XML specification 
sim.i.phy.nex         # Nexus file with character states annotated on tree
sim.i.json            # MASTER simulated output JSON log
sim.i.beast.log       # BEAST log with MASTER runtime messages (helps w/ debugging)
```
Some MASTER files are zipped as `.gz` when `sim_logging=compressed` or deleted when `sim_logging=clean`.
