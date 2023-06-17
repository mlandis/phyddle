# networks

The `network` directory contains subdirectories of trained networks, training results, and data re-normalization values. Subdirectories represent different pipeline projects.

phyddle will save the trained TensorFlow model, which can then be re-loaded later for additional predictions:
```
network/example/sim_batchsize128_numepoch20_nt200.hdf5
```

Three other files are needed to make predictions. Two contain renormalization factors (mean, sd) from the training dataset applied to new label and auxiliary data inputs, and the third contains conformalized quantile regression interval adjustments from the calibration procedure:

```
network/example/sim_batchsize128_numepoch20_nt200.cqr_interval_adjustments.csv
network/example/sim_batchsize128_numepoch20_nt200.train_label_norm.csv
network/example/sim_batchsize128_numepoch20_nt200.train_summ_stat_norm.csv
```

The `train_history` file logs learning metrics, such as loss score or mean square error, for the training and validation datasets across training epochs, and can be used to diagnose model overfitting:
```
network/example/sim_batchsize128_numepoch20_nt200.train_history.csv
```

The following `labels` and `pred` files report how well the neural network predicts (`pred.csv`) the true parameter for each dataset (`labels.csv`). The `pred_nocalib.csv` file uses conformalized quantile regression to compute prediction intervals that are *not* calibrated and often do not possess desired coverage properties. Comparing the prediction accuracy between the `test` and `train` datasets can also be used to diagnose model overfitting:
```
network/example/sim_batchsize128_numepoch20_nt200.test_labels.csv
network/example/sim_batchsize128_numepoch20_nt200.test_pred.csv
network/example/sim_batchsize128_numepoch20_nt200.test_pred_nocalib.csv
network/example/sim_batchsize128_numepoch20_nt200.train_labels.csv
network/example/sim_batchsize128_numepoch20_nt200.train_pred.csv
network/example/sim_batchsize128_numepoch20_nt200.train_pred_nocalib.csv
```
