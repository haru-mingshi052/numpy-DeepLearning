#==========================================
# SGD
#==========================================
class SGD:
    def __init__(self, lr):
        self.lr = lr
        
    def update(self, layers_dict):
        for layer in layers_dict.values():
            # weightの更新
            if hasattr(layer, 'weight'):
                layer.weight -= self.lr * layer.dw
               
            # biasの更新
            if hasattr(layer, 'bias'):
                layer.bias -= self.lr * layer.db

            # gammaの更新
            if hasattr(layer, 'gamma'):
              layer.gamma -= self.lr * layer.dgamma

            # betaの更新
            if hasattr(layer, 'beta'):
              layer.beta -= self.lr * layer.dbeta