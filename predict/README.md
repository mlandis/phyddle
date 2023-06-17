# Predictions

The `predict` directory stores prediction results for new datasets in various project subdirectories.

The new dataset should have one tree file (`.tre`) and one character matrix file (`.dat.nex`) that share a prefix name (e.g. `new.1`):

```
predict/example/new.1.tre
predict/example/new.dat.nex
```

These files can then be passed through Formatting to encode a phylo-state tensor, an auxiliary data tensor (contains summ. stats), and an extant version of the provided tree (if needed). The Python code for this would be:

```python
import Formatting
my_mdl = ModelLoader.load_model(my_args)
my_fmt = Formatting.Formatter(my_args, my_mdl)
pred_prefix = f"{my_args['pred_dir']}/{my_args['proj']}/{my_args['pred_prefix']}"
my_fmt.encode_one(tmp_fn=pred_prefix, idx=-1, save_phyenc_csv=True)
```

to create:
```
predict/example/new.1.info.csv
predict/example/new.1.summ_stat.csv 
predict/example/new.1.extant.tre     # if 'tree_type'=='extant'
predict/example/new.1.cdvs.csv       # if 'tree_type'=='extant'
predict/example/new.1.cblvs.csv      # if 'tree_type'=='serial'
```

The actual predictions produced by Predicting will output as a simple comma-separated value file:

```
predict/example/new.1.sim_batchsize128_numepoch20_nt500.pred_labels.csv
```
