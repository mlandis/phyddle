# Tensor data

This directory contains formatted versions of the raw data for tensor datasets.

Subdirectories represent different pipeline projects.

Raw datasets are categorized into different "tree-size" classes, which defines the number of columns in the compact phylo-state tensors. Possible tree-size classes are defined by the `'tree_size'` setting. Each individual dataset (a tree and character matrix pair) describing $n$ species is assigned to the largest tree-size class $m$ such that $n <= m$.

Categorized raw datasets are then formatted are stored either as gzip-compressed `.hdf5` format (default) or simple, uncompressed `.csv` table format. Compression typically results in ~20-30x smaller storage sizes, but cannot be read and parsed as simple text.

For example, if `'tree_size' == [100, 1000, 1000]` and `'tensor_format' == 'hdf5'` then the project directory might contain:
```
tensor_data/my_project/sim.nt100.hdf5
tensor_data/my_project/sim.nt1000.hdf5
tensor_data/my_project/sim.nt10000.hdf5
```

Alternatively, if `'tree_size' == [200, 500]` and `'tensor_format' == 'csv'` then the project directory might contain:
```
tensor_data/my_project/sim.nt200.cdvs.csv
tensor_data/my_project/sim.nt200.summ_stat.csv
tensor_data/my_project/sim.nt200.labels.csv
tensor_data/my_project/sim.nt500.cdvs.csv
tensor_data/my_project/sim.nt500.summ_stat.csv
tensor_data/my_project/sim.nt500.labels.csv
```
