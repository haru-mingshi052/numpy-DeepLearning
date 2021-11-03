import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dnn_processing import mnist_dataset, DnnBatchDataset
from dnn import DNN
from optimizer import SGD
from loss import SoftmaxWithLoss
from utility import TrainModel

import argparse

parser = argparse.ArgumentParser(
    description = "parameter for mnist training"
)

parser.add_argument("--layer_size", default=[200, 150], nargs="+", type=int,
                    help="隠れ層のサイズ")
parser.add_argument("--batch_size", default=128, type=int,
                    help="batchサイズの大きさ")
parser.add_argument("--epochs", default=300, type=int,
                    help="学習を何エポック回すか")
parser.add_argument("--es_patience", default=20, type=int,
                    help="何エポックスコアの改善が無かった時に学習を止めるか")
parser.add_argument("--seed", default=71, type=int,
                    help='シード値')

args = parser.parse_args()

import warnings
warnings.filterwarnings('ignore')

# layer_sizeの先頭と終わりに層の大きさを追加
args.layer_size.insert(0, x_train.shape[1])
args.layer_size.append(y_train.shape[1])

if __name__ == "__main__":
    x_train, x_test, y_train, y_test = mnist_dataset()
    trainer = TrainModel(
        model=DNN(layer_size=args.layer_size),
        loss_layer=SoftmaxWithLoss(),
        optimizer=SGD(args.learning_rate),
        epochs=args.epochs,
        es_patience=args.es_patience,
        train_ds=DnnBatchDataset(x_train, y_train, args.batch_size),
        val_ds=DnnBatchDataset(x_test, y_test, args.batch_size),
        seed=args.seed
    )
    trainer.train()