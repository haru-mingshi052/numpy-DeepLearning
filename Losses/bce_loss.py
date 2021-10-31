import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import cupy as cp

from utility import Layer
from .functions import binary_cross_entropy

class BCELoss(Layer):
    def __init__(self):
        self.y = None
        self.t = None
        self.loss = None
        
    def forward(self, y, t):
        self.t = t # target変数
        self.y = y # 予測
        self.loss = binary_cross_entropy(self.y, self.t) # 損失を計算
        
        return self.loss
    
    def backward(self, dout):
        delta = 1e-7
        dx = self.t * dout * -1 / (self.y + delta) + (1 - self.t) * dout / (1 - self.y + delta)
        return dx