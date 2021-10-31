#==========================================
# SGD
#==========================================
class SGD:
    def __init__(self, lr):
        self.lr = lr
        self.update_params = {'weight': 'dw', 'bias': 'db', 'gamma': 'dgamma', 'beta': 'dbeta'}

    def update(self, layers_dict):
        for layer in layers_dict.values():
            for key, value in self.update_params.items():
                if hasattr(layer, key):
                    param = getattr(layer, key)
                    gradient = getattr(layer, value)
                    param -= self.lr * gradient
                    setattr(layer, key, param)