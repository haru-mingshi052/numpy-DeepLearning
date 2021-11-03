import numpy as np
import cupy as cp
import time

class TrainModel:
    def __init__(self, model, loss_layer, optimizer, epochs, es_patience, train_ds, val_ds, seed):
        self.model = model 
        self.loss_layer = loss_layer
        self.optimizer = optimizer
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.epochs = epochs
        self.es_patience = es_patience
        self.train_loss_list = []
        self.val_loss_list = []
        self.seed = seed

    def seed_everything(self, seed):
        np.random.seed(seed)
        cp.random.seed(seed)

    def train(self):
        best_loss = np.inf
        patience = self.es_patience
        self.seed_everything(self.seed)

        print("==========start==========")
        for epoch in range(self.epochs+1):
            start_time = time.time()
            train_loss = 0.0
            val_loss = 0.0
        
            # train
            train_loss = self.loop(
                ds=self.train_ds,
                train_flg=True
            )
            
            # eval
            val_loss = self.loop(
                ds=self.val_ds,
                train_flg=False
            )
                
            finish_time = time.time()
            
            self.train_loss_list.append(train_loss)
            self.val_loss_list.append(val_loss)
            
            # 10 epochごとに結果を表示
            if epoch % 10 == 0:
                print("Epochs：{:03} | Train Loss：{:.5f} | Val Loss：{:.5f} | Time：{:.3f}"
                    .format(epoch, train_loss, val_loss, finish_time - start_time))
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                patience = self.es_patience
            else:
                patience -= 1
        
            if patience == 0:
                print("Early Stopping | Epochs：{:03} | Best Loss：{:.7f}"
                    .format(best_epoch, best_loss))
                break

    def loop(self, ds, train_flg):
        epoch_loss = 0.0
        for iter_num in range(ds.iter_nums):
            x, y = ds.__getitem__()
                
            # numpyからcupyへ
            x = cp.asarray(x)
            y = cp.asarray(y)
                
            z = self.model.forward(x) # 順伝播
            loss = self.loss_layer.forward(z, y) # lossの計算

            loss = cp.asnumpy(loss)
            epoch_loss += loss

            if train_flg:
                # 逆伝播
                dx = self.loss_layer.backward(dout = 1)
                self.model.backward(dx)
                
                self.optimizer.update(self.model.layers) # パラメータの更新
        
        epoch_loss /= ds.iter_nums

        return epoch_loss