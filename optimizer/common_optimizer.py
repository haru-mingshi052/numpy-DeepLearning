#=================
# 継承
#=================
class Optimizer:
    def __init__(self, lr, update_params):
        self.lr = lr
        if update_params==None:
            self.update_params = {'weight': 'dw', 'bias': 'db', 'gamma': 'dgamma', 'beta': 'dbeta'}
        else:
            self.update_params=update_params

    def update(self, layers_dict):
        for layer in layers_dict.values():
            for key, value in self.update_params.items():
                if hasattr(layer, key):
                    param = getattr(layer, key)
                    gradient = getattr(layer, value)
                    new_param = expression(lr, param, gradient)
                    setattr(layer, key, new_param)

    def expression(self, lr, param, gradient):
        raise NotImplementedError()