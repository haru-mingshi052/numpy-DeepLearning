import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import cupy as cp

from utility import Layer
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