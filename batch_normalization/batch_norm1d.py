from .base_batchnorm import BaseBatchNormalization

class BatchNorm1d(BaseBatchNormalization):
    def __init__(self, gamma=1.0, beta=0.0, momentum=0.9, running_mean=None, running_var=None):
        super().__init__(gamma, beta, momentum, running_mean, running_var)
        
    def forward(self, x, train):
        self.input_shape = x.shape

        out = self.forward_(x, train)
        out.reshape(*self.input_shape)
        
        return out
    
    def backward(self, dout):
        dx = self.backward_(dout)
        dx.reshape(*self.input_shape)
        
        return dx