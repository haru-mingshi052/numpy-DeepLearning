from .common_optimizer import Optimizer

class SGD(Optimizer):
    def __init__(self, lr, update_params=None):
        super().__init__(lr, update_params)

    def update(self, layers_dict):
        for layer in layers_dict.values():
            for key, value in self.update_params.items():
                if hasattr(layer, key):
                    param = getattr(layer, key)
                    gradient = getattr(layer, value)
                    param -= self.lr * gradient
                    setattr(layer, key, param)