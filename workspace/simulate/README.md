# Raw data

The `raw_data` directory contains directories of simulated training replicates for each model generated by the Simulating step. In some cases, it includes output from Formatting, as well. Subdirectories represent different pipeline projects.

The Simulating step will create the following files for simulation replicate `i`:
```
raw_data/example/sim.i.tre             # Newick phylogeny string
raw_data/example/sim.i.dat.nex         # Nexus character matrix
raw_data/example/sim.i.param_row.csv   # simulation parameters with one parameter per row
raw_data/example/sim.i.param_col.csv   # simulation parameters with one parameter per column
```

The Formatting step additionally creates the following files when preparing the encoded tensor:
```
raw_data/example/sim.i.info.csv        # information about replicate settings
raw_data/example/sim.i.extant.tre      # Newick phylogeny string with only extant taxa
                                       # when 'tree_type' == 'extant'
```

Phylogenetic-state tensor encodings are also stored only when `'save_phyenc_csv' == True`:
```
raw_data/example/sim.i.cdvs.tre        # compact diversity vector + brlen info expansion + states
                                       # when 'tree_type' == 'extant'
raw_data/example/sim.i.cblvs.tre       # compact bijective ladderized vector + brlen info expansion + states
                                       # when 'tree_type' == 'serial')
```

When MASTER is used to simulate data, the follow additional files may appear:
```
raw_data/example/sim.i.xml             # MASTER XML specification 
raw_data/example/sim.i.phy.nex         # Nexus file with character states annotated on tree
raw_data/example/sim.i.json            # MASTER simulated output JSON log
raw_data/example/sim.i.beast.log       # BEAST log with MASTER runtime messages (helps w/ debugging)
```
Some MASTER files are zipped as `.gz` when `sim_logging=compressed` or deleted when `sim_logging=clean`.