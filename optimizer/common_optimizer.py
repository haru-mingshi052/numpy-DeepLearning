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

    def forward(self, layers_dict):
        raise NotImplementedError