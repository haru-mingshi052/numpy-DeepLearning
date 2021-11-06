from .common_optimizer import Optimizer

class SGD(Optimizer):
    def __init__(self, lr, update_params=None):
        super().__init__(lr, update_params)

    def expression(lr, param, gradient):
        return param - self.lr * gradient