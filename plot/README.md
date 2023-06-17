# plots

The `plot` directory contains subdirectories for different projects with figures produced by the Plotting step.

The `summary.pdf` contains all figures produced by Plotting. The prefix for the figure names corresponds to the trained network name.

Training history plots are report how training metrics (loss, MSE, etc.) changed across epochs for training and validation datasets. The `history_.pdf` figure reports the loss for the entire network during training, while other history files report all training metrics for that particular prediction target's output layer.

```
plot/my_project/sim_batchsize128_numepoch20_nt200_history_.pdf
plot/my_project/sim_batchsize128_numepoch20_nt200_history_param_value.pdf
plot/my_project/sim_batchsize128_numepoch20_nt200_history_param_lower.pdf
plot/my_project/sim_batchsize128_numepoch20_nt200_history_param_upper.pdf
```

Scatterplots for prediction accuracy for test and train datasets are generated for each prediction target (e.g. parameter of interest). Scatterplots portray the point estimate and intervals correspond to calibrated prediction intervals. If the predictions differ substantially between train and test examples, it may indicate your network is undertrained or overtrained.

```
plot/my_project/sim_batchsize128_numepoch20_nt200_train_b_0_1.pdf
plot/my_project/sim_batchsize128_numepoch20_nt200_train_d_0_1.pdf
plot/my_project/sim_batchsize128_numepoch20_nt200_train_w_0.pdf
plot/my_project/sim_batchsize128_numepoch20_nt200_train_e_0.pdf
plot/my_project/sim_batchsize128_numepoch20_nt200_test_b_0_1.pdf
plot/my_project/sim_batchsize128_numepoch20_nt200_test_d_0_1.pdf
plot/my_project/sim_batchsize128_numepoch20_nt200_test_w_0.pdf
plot/my_project/sim_batchsize128_numepoch20_nt200_test_e_0.pdf
```

The density (histogram) plots summarize the extent of your training dataset and labels.
Densities are plotted for all training input auxiliary data (summary statistics and "known" parameters) and for labels (model parameters). In addition, PCA for the input training auxiliary data is plotted. When a prediction dataset is provided in the settings file, Plotting will include information about that new dataset's value with respect to the densities and PCA, typically indicated with a red line, dot, and text. If the new dataset is an extreme value with represent to the training dataset, it suggests the network will not make good predictions.
