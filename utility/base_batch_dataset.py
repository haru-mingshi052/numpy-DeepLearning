import numpy as np

class BaseBatchDataset:
    def __init__(self, X, y, batch_size, iter_nums=None):
        self.feature = X # 説明変数
        self.target = y # 目的変数
        self.batch_mask = np.random.choice(len(self.feature), batch_size)
        if iter_nums==None:
            self.iter_nums = len(self.feature) // batch_size
        else:
            self.iter_nums = iter_nums # 1epochで何イテレーション回すか
    
    def __getitem__(self):
        x_batch, y_batch = self.batch_choice()
        
        x_batch = np.array(x_batch)
        y_batch = np.array(y_batch)

        return x_batch, y_batch
    
    def batch_choice(self):
        raise NotImplementedError()