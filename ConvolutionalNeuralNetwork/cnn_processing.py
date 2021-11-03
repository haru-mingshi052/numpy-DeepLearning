import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from keras.datasets import cifar10
from keras.utils import np_utils

from utility import BaseBatchDataset

class CnnBatchDataset(BaseBatchDataset):
    def batch_choice(self):
        x_batch = []
        y_batch = []
        
        for i in self.batch_mask:
            x_batch.append(self.feature[i] / 255.0)
            y_batch.append(self.target[i])

        return x_batch, y_batch

def cifar10_dataset():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.transpose(0, 3, 1, 2)
    x_test = x_test.transpose(0, 3, 1, 2)
    y_train = np_utils.to_categorical(y_train,10)
    y_test = np_utils.to_categorical(y_test,10)

    return x_train, x_test, y_train, y_test