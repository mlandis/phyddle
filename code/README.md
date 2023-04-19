# phyddle - code

This directory contains code to simulate data, prepare it into tensor format, and then use it to train neural networks and plot diagnostics.

Example use:
```
# execute from code subdirectory
cd ~/projects/phyddle/code

# remove all previous dirs for model name `bd1`
./clean_model.sh --name bd1

# simulate data in `raw_data/bd1`
./simulate_training_data.py --name bd1 --start_idx 0 --end_idx 100 --use_parallel

# prepare data as tensors in `formatted_data/bd1`
./prepare_training_data.py --name bd1

# train CNN using tensors in `network/bd1`
./train_rates_cnn.py --name bd1 --num_epoch 20 --batch_size 32 --num_validation 50 --num_test 50 --max_taxa 500
```
