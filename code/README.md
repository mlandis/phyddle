# phyddle - code

This directory contains code to simulate data, prepare it into tensor format, and then use it to train neural networks and plot diagnostics.

Example use:
```
# execute from code subdirectory
cd ~/projects/phyddle/code

# remove all previous dirs for model name `my_job`
./clean_model.sh --name my_job

# config file controls pipeline
# (will add more argparse options)
vim config.py

# simulate data in `raw_data/my_job`
./run_simulate.py --cfg config

# prepare data as tensors in `tensor_data/my_job`
./run_format.py --cfg config

# train CNN using tensors in `network/my_job`
./run_learn.py --cfg config
```
