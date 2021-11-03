import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from activation import Relu
from batch_normalization import BatchNorm2d
from dropout import Dropout
from layer import Conv2d, AveragePooling2d, MaxPooling2d, Affine
from utility import Model

class CNN(Model):
    def __init__(self):
        super().__init__()

        # block1
        self.layers['conv2d1-1'] = Conv2d(3, 64, 3, 3, 1, 1)
        self.layers['relu1-1'] = Relu()
        self.layers['batch_norm-1'] = BatchNorm2d()
        self.layers['dropout-1'] = Dropout(p = 0.2)
        
        self.layers['conv2d1-2'] = Conv2d(64, 64, 3, 3, 1, 1)
        self.layers['relu1-2'] = Relu()
        self.layers['batch_norm-2'] = BatchNorm2d()
        self.layers['maxpool-1'] = MaxPooling2d(2, 2, 2, 0)
        
        # block2
        self.layers['conv2d2-1'] = Conv2d(64, 128, 3, 3, 1, 1)
        self.layers['relu2-1'] = Relu()
        self.layers['batch_norm-3'] = BatchNorm2d()
        self.layers['dropout-2'] = Dropout(p = 0.2)
        
        self.layers['conv2d2-2'] = Conv2d(128, 128, 3, 3, 1, 1)
        self.layers['relu2-2'] = Relu()
        self.layers['batch_norm-4'] = BatchNorm2d()
        self.layers['maxpool-2'] = MaxPooling2d(2, 2, 2, 0)
        
        # block3
        self.layers['conv2d3-1'] = Conv2d(128, 256, 3, 3, 1, 1)
        self.layers['relu3-1'] = Relu()
        self.layers['batch_norm-5'] = BatchNorm2d()
        self.layers['dropout-3'] = Dropout(p = 0.2)
        
        self.layers['conv2d3-2'] = Conv2d(256, 256, 3, 3, 1, 1)
        self.layers['relu3-2'] = Relu()
        self.layers['batch_norm-6'] = BatchNorm2d()
        self.layers['dropout-4'] = Dropout(p = 0.2)
        
        # last block
        self.layers['conv2d3-3'] = Conv2d(256, 256, 3, 3, 1, 1)
        self.layers['GAP'] = AveragePooling2d(8, 8, 8, 0)
        self.layers['Affine'] = Affine(256, 10)