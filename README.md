# phyddle

Python scripts for using CNNs + deep learning to estimate SSE model parameters.

The repo has two main directories. The code directory contains scripts to simulate and train networks under a given model. The model directory contains the data and trained networks for each considered model.

Example use:
cd ~/projects/phyddle/code
./simulate_training_data.py --start_idx 0 --end_idx 100 --use_parallel --name geosse_share_v1
./prepare_training_data.py --name geosse_share_v1
./train_rates_cnn.py --name geosse_share_v1 --num_epoch 20 --batch_size 32 --num_validation 50 --num_test 50
