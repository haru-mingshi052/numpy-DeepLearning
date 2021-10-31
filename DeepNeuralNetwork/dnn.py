import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from collections import OrderedDict

from layer import Affine
from activation import Relu
from batch_normalization import BatchNorm1d
from dropout import Dropout

#================================
# DNN
#================================
class DNN:
  def __init__(self, layer_size, sigmoid_flg = False):
    self.layers = OrderedDict()
    for i in range(len(layer_size) - 2):
      self.layers[f"affine{i}"] = Affine(layer_size[i], layer_size[i + 1])
      self.layers[f"relu{i}"] = Relu()
      self.layers[f"batch_norm{i}"] = BatchNorm1d(gamma = 1, beta = 0)
      self.layers[f"dropout{i}"] = Dropout(0.5)

    self.layers["affine_last"] = Affine(layer_size[len(layer_size) - 2], layer_size[len(layer_size) - 1])

    if sigmoid_flg:
      self.layers['sigmoid'] = sigmoid()

  def forward(self, x, train = True):
    for key, layer in self.layers.items():
        if 'dropout' in key or 'batch_norm' in key:
            x = layer.forward(x, train)
        else:
            x = layer.forward(x)
    
    return x

  def backward(self, dout):
    layers = list(self.layers.values())
    layers.reverse()
    for layer in layers:
        dout = layer.backward(dout)