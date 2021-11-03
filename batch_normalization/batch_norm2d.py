from .base_batchnorm import BaseBatchNormalization

class BatchNorm2d(BaseBatchNormalization):
    def __init__(self, gamma=1.0, beta=0.0, momentum=0.9, running_mean=None, running_var=None):
        super().__init__(gamma, beta, momentum, running_mean, running_var)
        
    def forward(self, x, train):
        self.input_shape = x.shape
        
        N, C, H, W = x.shape
        x = x.reshape(N, -1)
        
        out = self.forward_(x, train)
        
        return out.reshape(*self.input_shape)
    
    
    def backward(self, dout):
        N, C, H, W = dout.shape
        dout = dout.reshape(N, -1)
        
        dx = self.backward_(dout)
        
        return dx.reshape(*self.input_shape)