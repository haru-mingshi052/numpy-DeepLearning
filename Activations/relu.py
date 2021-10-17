import cupy as cp

from ..Utility.abstract_layer import Layer
from .functions import relu

class Relu(Layer):
    def __init__(self):
        self.mask = None
        
    def forward(self, x):
        out, self.mask = relu(x)
        
        return out
    
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        
        return dx