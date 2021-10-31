from collections import OrderedDict

class Model:
    def __init__(self):
        self.layers = OrderedDict()

    def forward(self, x, train = True):
        for key, layer in self.layers.items():
            if 'dropout' in key or 'batch_norm' in key:
                x = layer.forward(x, train)
            else:
                x = layer.forward(x)
                
        return x

    def backward(self, dout):
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)