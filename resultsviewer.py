import numpy as np

data_pred = np.load("results/CRSP_large_exp0_0_pred.npy")
data_true = np.load("results/CRSP_large_exp0_0_true.npy")

print(data_pred.shape)
print("predicted")
print(data_pred[:3, 0])
print("true")
print(data_true[:3, 0])