import h5py
import os
from utils import bo
from utils import *
from train_sugarbyte import train, locate_previous_epochs
import time
import numpy as np
import pandas as pd


h5_train = h5py.File('data/proc/cloud_train.hdf5', 'r')
h5_validate = h5py.File('data/proc/cloud_val.hdf5', 'r')


print("{} : (debugging)".format(__file__))
model_directory = "data/models/debug"
last_checkpoint = locate_previous_epochs(model_directory)

os.makedirs(model_directory, exist_ok=True)

# Here
params = bo.SugarbyteParams(
    dilation=1,
    kernel_shrink=-1,
    depth=3,
    batch_size=4,
    directory=model_directory,
    kernel_size=3,
    filters=8,
    epochs=9,
    train_h5=h5_train['images'],
    validate_h5=h5_validate['images'],
    test_h5=None
)

params.validate()
# loss = train(params, small_dataset=True)
loss, acc = train(params, load=last_checkpoint, small_dataset=True)
