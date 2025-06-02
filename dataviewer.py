import pickle
import numpy as np
from util import *
import h5py

data = load_dataset("data/METR-LA", 64, 64, 64)
print(data.keys())
print("x")
print(data['x_train'].shape)
# (number observations, window size, number sensors, (observed value, timestamp 5 mins/day = 0.00347222))



print(data['x_train'][0, :, 0])


f = h5py.File('data/metr-la.h5', 'r')
# axis0, block0_item sensor label, axis1 timestamp, block0_values the actual values

dataset = f['df']
print(dataset.keys())
print(dataset['block0_values'].shape)
print(dataset['block0_values'][:12, 0])
