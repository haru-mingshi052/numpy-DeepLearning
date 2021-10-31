import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from layer import Affine
from activation import Relu
from batch_normalization import BatchNorm1d
from dropout import Dropout
from utility import Model

#================================
# DNN
#================================
class DNN(Model):
  def __init__(self, layer_size, sigmoid_flg = False):
    super().__init__()

    for i in range(len(layer_size) - 2):
      self.layers[f"affine{i}"] = Affine(layer_size[i], layer_size[i + 1])
      self.layers[f"relu{i}"] = Relu()
      self.layers[f"batch_norm{i}"] = BatchNorm1d(gamma = 1, beta = 0)
      self.layers[f"dropout{i}"] = Dropout(0.5)

    self.layers["affine_last"] = Affine(layer_size[len(layer_size) - 2], layer_size[len(layer_size) - 1])

    if sigmoid_flg:
      self.layers['sigmoid'] = sigmoid()