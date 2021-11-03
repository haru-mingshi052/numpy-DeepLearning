import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import cupy as cp

from utility import Layer, im2col, col2im

class Conv2d(Layer):
    def __init__(self, in_channel, out_channel, filter_h, filter_w, stride, padding):
        self.in_channel = in_channel # 入力チャンネル数
        self.out_channel = out_channel # 出力チャンネル数
        self.filter_h = filter_h # filter height
        self.filter_w = filter_w # filter width
        self.stride = stride # ストライド幅
        self.padding = padding # パディング
        
        # パラメータ
        self.weight = cp.random.randn(out_channel, in_channel, filter_h, filter_w) * 0.01
        self.bias = cp.random.randn(out_channel) * 0.01
        
        # 中間データ
        self.col_x = None
        self.col_w = None
        self.x = None
        
        # 勾配
        self.dw = None
        self.db = None
        
    def forward(self, x):
        N, C, H, W = x.shape
        
        # 出力サイズの計算
        out_h = int(1 + (H + 2 * self.padding - self.filter_h) / self.stride) 
        out_w = int(1 + (W + 2 * self.padding - self.filter_w) / self.stride)
            
        self.col_x = im2col(x, self.filter_h, self.filter_w, self.stride, self.padding)
        self.col_w = self.weight.reshape(self.out_channel, self.in_channel * self.filter_h * self.filter_w).T
            
        out = cp.dot(self.col_x, self.col_w) + self.bias
            
        out = out.reshape(N, out_h, out_w, self.out_channel).transpose(0, 3, 1, 2)
        
        self.x = x # 入力を変数として保持
            
        return out
    
    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, self.out_channel)
        
        # バイアスの勾配
        self.db = cp.sum(dout, axis = 0)
        
        # 重みの勾配
        self.dw = cp.dot(self.col_x.T, dout)
        self.dw = self.dw.transpose(1, 0).reshape(self.out_channel, self.in_channel, self.filter_h, self.filter_w)
        
        # 次の層に逆伝播する値
        dx = cp.dot(dout, self.col_w.T)
        dx = col2im(dx, self.x.shape, self.filter_h, self.filter_w, self.stride, self.padding)
        
        return dx