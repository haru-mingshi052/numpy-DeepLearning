import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import cupy as cp

from .base_layer import BasePooling
from utility import im2col, col2im

class MaxPooling2d(BasePooling):
    def __init__(self, filter_h=2, filter_w=2, stride=2, padding=0):
        super().__init__(filter_h, filter_w, stride, padding)
        
        self.arg_max = None
        self.x = None
        
    def forward(self, x):
        N, C, H, W = x.shape
        
        # 出力サイズの計算
        OH = int(1 + (H - self.filter_h) / self.stride)
        OW = int(1 + (W - self.filter_w) / self.stride)
        
        col = im2col(x, self.filter_h, self.filter_w, self.stride, self.padding)
        col = col.reshape(N * OH * OW * C, self.filter_h * self.filter_w)
        
        self.arg_max = cp.argmax(col, axis=1)
        out = cp.max(col, axis=1).reshape(N, OH, OW, C).transpose(0, 3, 1, 2)
        
        self.x = x # 入力を変数として保持
        
        return out
    
    def backward(self, dout):
        N, C, OH, OW = dout.shape
        dout = dout.transpose(0, 2, 3, 1)
        
        dx = cp.zeros((dout.size, self.filter_h * self.filter_w))
        dx[cp.arange(self.arg_max.size), self.arg_max] = dout.flatten()
        dx = dx.reshape(N * OH * OW, C * self.filter_h * self.filter_w)
        dx = col2im(dx, self.x.shape, self.filter_h, self.filter_w, self.stride, self.padding)
        
        return dx