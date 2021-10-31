import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import cupy as cp
import time

from processing import mnist_dataset
from dnn import DNN
from optimizers import SGD
from losses import SoftmaxWithLoss

import argparse

parser = argparse.ArgumentParser(
    description = "parameter for mnist training"
)

parser.add_argument("--layer_size", default = [200, 150], nargs = "+", type = int,
                    help = "隠れ層のサイズ")
parser.add_argument("--batch_size", default = 128, type = int,
                    help = "batchサイズの大きさ")
parser.add_argument("--epochs", default = 300, type = int,
                    help = "学習を何エポック回すか")
parser.add_argument("--es_patience", default = 20, type = int,
                    help = "何エポックスコアの改善が無かった時に学習を止めるか")

args = parser.parse_args()

import warnings
warnings.filterwarnings('ignore')

np.random.seed(71)

x_train, x_val, y_train, y_val = mnist_dataset() # データの準備

# layer_sizeの先頭と終わりに層の大きさを追加
args.layer_size.insert(0, x_train.shape[1])
args.layer_size.append(y_train.shape[1])

# モデルの作成
model = DNN(
    layer_size = args.layer_size
)

#=====================================
# train_model
#=====================================
def train_model(model, x_train, y_train, x_val, y_val, batch_size, epochs, es_patience):
    # 変数の設定
    train_size = x_train.shape[0]
    val_size = x_val.shape[0]
    tr_iter_nums = train_size // batch_size # 1epochで何イテレーション回すか
    val_iter_nums = val_size // batch_size # 1epochで何イテレーション回すか
    learning_rate = 1e-3 # 学習率
    best_loss = np.inf # lossを管理する変数最初はinfに
    patience = es_patience # early stoppingを管理する変数
        
    train_loss_list = []
    val_loss_list = []
        
    loss_layer = SoftmaxWithLoss() # loss
    optimizer = SGD(lr = learning_rate) # optimizer
       
    #==================================
    # 学習ループスタート
    #==================================
    print('=========start==========')
    for epoch in range(epochs + 1):
        start_time = time.time()
        train_loss = 0.0
        val_loss = 0.0

        # train
        for iter_num in range(tr_iter_nums):
            batch_mask = np.random.choice(train_size, batch_size)
            x_batch = x_train[batch_mask]
            y_batch = y_train[batch_mask]

            x_batch = cp.asarray(x_batch)
            y_batch = cp.asarray(y_batch)
                
            # 順伝播
            z = model.forward(x_batch)
            loss = loss_layer.forward(z, y_batch)

            loss = cp.asnumpy(loss)
                
            train_loss += loss
                
            # 逆伝播
            dx = loss_layer.backward(dout = 1)
            model.backward(dx)
                
            optimizer.update(model.layers)
        
        train_loss /= tr_iter_nums
                
        # eval
        for iter_num in range(val_iter_nums):
            batch_mask = np.random.choice(val_size, batch_size)
            x_batch = x_val[batch_mask]
            y_batch = y_val[batch_mask]

            x_batch = cp.asarray(x_batch)
            y_batch = cp.asarray(y_batch)

            z = model.forward(x_batch) # 順伝播
            loss = loss_layer.forward(z, y_batch) # lossの計算

            loss = cp.asnumpy(loss)

            val_loss += loss

        val_loss /= val_iter_nums

        finish_time = time.time()

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        
        # 10 epochごとに結果を表示
        if epoch % 10 == 0:
            print("Epochs：{:03} | Train Loss：{:.5f} | Val Loss：{:.5f} | Time：{:.3f}"
                .format(epoch, train_loss, val_loss, finish_time - start_time))
        
        # early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            patience = es_patience
        else:
            patience -= 1
        
        if patience == 0:
            print("Early Stopping | Epoch：{:03} | Best Loss：{:.5f}".format(best_epoch, best_loss))
            break

if __name__ == "__main__":
    train_model(
        model, 
        x_train, 
        y_train,
        x_val,
        y_val,
        batch_size = args.batch_size,
        epochs = args.epochs,
        es_patience = args.es_patience
    )