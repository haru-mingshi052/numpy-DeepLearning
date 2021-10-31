import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import cupy as cp

from utility import Layer

#=============================================
# Dropout
#=============================================
class Dropout(Layer):
    def __init__(self, p):
        self.p = p
        self.mask = None
        
    def forward(self, x, train = True):
        if train:
            self.mask = cp.random.rand(*x.shape) > self.p
            return x * self.mask
        else:
            return x * (1.0 - self.p)
        
    def backward(self, dout):
        return dout * self.mask