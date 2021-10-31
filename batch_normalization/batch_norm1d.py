import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import cupy as cp

from utility import Layer

#================================================
# BatchNormalization
#================================================
class BatchNorm1d(Layer):
    def __init__(self, gamma, beta, momentum = 0.9, running_mean = None, running_var = None):
        #更新するパラメータ
        self.gamma = gamma #γ
        self.beta = beta #β
        
        #計算に使うパラメータ
        self.momentum = momentum
        self.input_shape = None
        
        #テスト時に使用する平均と分散
        self.running_mean = running_mean
        self.running_var = running_var
        
        #backwardに使用するデータ
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None
        
    def forward(self, x, train):
        self.input_shape = x.shape
        
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = cp.zeros(D)
            self.running_var = cp.zeros(D)
            
        if train:
            mu = x.mean(axis = 0) #平均 μ
            xc = x - mu
            var = cp.mean(xc**2, axis = 0) #分散 σ^2
            std = cp.sqrt(var + 1e-6) #標準偏差 σ
            xn = xc / std
            
            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
            
        else:
            xc = x - self.running_mean
            xn = xc / ((cp.sqrt(self.running_var + 10e-7)))

        out = self.gamma * xn + self.beta
        
        return out.reshape(*self.input_shape)
    
    
    def backward(self, dout):
        dbeta = dout.sum(axis = 0)
        dgamma = cp.sum(self.xn * dout, axis = 0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -cp.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = -cp.sum(dxc, axis = 0)
        dx = dxc + dmu / self.batch_size
        
        self.dgamma = dgamma
        self.dbeta = dbeta
        
        return dx.reshape(*self.input_shape)