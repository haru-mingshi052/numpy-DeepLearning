import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import cupy as cp

from utility import Layer
from .functions import relu

# シグモイド関数のレイヤー
class Sigmoid(Layer):
    def __init__(self):
        self.out = None
        
    def forward(self, x):
        self.out = sigmoid(x)
        
        return self.out
    
    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        
        return dx