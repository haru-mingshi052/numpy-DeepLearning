import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from cnn import CNN
from cnn_processing import cifar10_dataset, CnnBatchDataset
from optimizer import SGD
from loss import SoftmaxWithLoss
from utility import TrainModel

import argparse

parser = argparse.ArgumentParser(
    description = "parameter for mnist training"
)

parser.add_argument("--batch_size", default=16, type=int,
                    help="batchサイズの大きさ")
parser.add_argument("--epochs", default=300, type=int,
                    help="学習を何エポック回すか")
parser.add_argument("--es_patience", default=20, type=int,
                    help="何エポックスコアの改善が無かった時に学習を止めるか")
parser.add_argument("--learning_rate", default=1e-3, type=float,
                    help="optimizerの学習率")
parser.add_argument("--seed", default=71, type=int,
                    help="シード値")

args = parser.parse_args()

import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
  x_train, x_test, y_train, y_test = cifar10_dataset()
  trainer = TrainModel(
    model=CNN(),
    loss_layer=SoftmaxWithLoss(),
    optimizer=SGD(args.learning_rate),
    epochs=args.epochs,
    es_patience=args.es_patience,
    train_ds=CnnBatchDataset(x_train, y_train, args.batch_size, iter_nums=100),
    val_ds=CnnBatchDataset(x_test, y_test, args.batch_size, iter_nums=100),
    seed=args.seed
  )
  trainer.train()