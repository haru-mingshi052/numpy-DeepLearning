import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utility import Layer

class BasePooling(Layer):
    def __init__(self, filter_h, filter_w, stride, padding):
        self.filter_h = filter_h # filter height
        self.filter_w = filter_w # filter width
        self.stride = stride # ストライド
        self.padding = padding # パディング  