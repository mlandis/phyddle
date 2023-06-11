# phyddle - code

This directory contains code to simulate data, prepare it into tensor format, and then use it to train neural networks and plot diagnostics.

Example use:
```
# execute from code subdirectory
cd ~/projects/phyddle/code

# remove all previous dirs for project name `my_project`
./clean_project.sh --proj my_project

# config file controls pipeline
# (edit dictionary values)
vim config.py

# simulate data in `raw_data/my_project`
./run_simulate.py --cfg config

# prepare data as tensors in `tensor_data/my_project`
./run_format.py --cfg config

# train CNN using tensors in `network/my_project`
./run_learn.py --cfg config

# predict parameters for new dataset using trained CNN
./run_predict.py --cfg config

# plot output into `plot/my_project`
./run_plot.py --cfg config

```
