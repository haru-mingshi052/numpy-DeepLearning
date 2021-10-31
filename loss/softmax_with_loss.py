import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import cupy as cp

from utility import Layer
from activation.functions import softmax
from .functions import cross_entropy_error

class SoftmaxWithLoss(Layer):
    def __init__(self):
        self.y = None
        self.t = None
        self.loss = None
        
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss
    
    def backward(self, dout = 1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx