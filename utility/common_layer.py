#=================
# 継承
#=================
class Layer:
    def forward(self, x):
        raise NotImplementedError

    def backward(self, dout):
        raise NotImplementedError
