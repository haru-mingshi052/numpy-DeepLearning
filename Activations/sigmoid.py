import cupy as cp

from ..Utility.abstract_layer import Layer
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