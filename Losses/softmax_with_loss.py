import cupy as cp

from ..Utility.abstract_layer import Layer
from ..Activations.functions import softmax
from ..function import cross_entropy_error

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