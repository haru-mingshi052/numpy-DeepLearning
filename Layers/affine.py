import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import cupy as cp

from utility import Layer

#===========================================
# Affine
#===========================================
class Affine(Layer):
    def __init__(self, input_size, output_size):
        self.weight = cp.random.randn(input_size, output_size)
        self.bias = cp.random.randn(output_size)

        self.x = None
        self.dw = None
        self.db = None

    def forward(self, x):
        self.x = x
        y = cp.dot(x, self.weight) + self.bias

        return y

    def backward(self, dout):
        dx = cp.dot(dout, self.weight.T)
        self.dw = cp.dot(self.x.T, dout)
        self.db = cp.sum(dout, axis = 0)

        return dx