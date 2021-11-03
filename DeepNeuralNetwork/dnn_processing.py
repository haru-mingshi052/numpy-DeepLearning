import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from utility import BaseBatchDataset

class DnnBatchDataset(BaseBatchDataset):
    """
    Dnn用のバッチデータセット
    """
    def batch_choice():
        x_batch = self.feature[self.batch_mask]
        y_batch = self.target[self.batch_mask]

        return x_batch, y_batch

def to_categorical(y, N):
    """
    targetをカテゴリ値にする関数
    args:
        y: target変数
        N: カテゴリ数
    return:
        new_y: カテゴリ値に変更したtarget変数
    """
    y = np.array([int(i) for i in y])
    new_y = np.zeros((y.shape[0], N), dtype = "float32")
    new_y[np.arange(len(y)), y] = 1

    return new_y

def mnist_dataset():
    """
    mnistデータの準備
    """
    x, y = datasets.fetch_openml('mnist_784', version = 1, return_X_y = True)
    x_train, x_val, y_train, y_val = train_test_split(x / 255, y, test_size = 0.2, random_state = 71)
    y_train = to_categorical(y_train, 10)
    y_val = to_categorical(y_val, 10)

    return x_train, x_val, y_train, y_val