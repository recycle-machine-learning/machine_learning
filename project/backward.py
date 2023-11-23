import torch
from collections import OrderedDict
from layers import *
class Backward:

    def __init__(self, model):
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu1 = model.relu1
        self.pool1 = model.pool1

        self.conv2 = model.conv2
        self.bn2 = model.bn2
        self.relu2 = model.relu2
        self.pool2 = model.pool2

        self.fc1 = model.fc1


    def backward(self, x):
        dout = self.fc1.backward(x)

        dout = self.pool2.backward(dout)
        dout = self.relu2.backward(dout)
        dout = self.bn2.backward(dout)
        dout = self.conv2.backward(dout)

        dout = self.pool1.backward(dout)
        dout = self.relu1.backward(dout)
        dout = self.bn1.backward(dout)
        dout = self.conv1.backward(dout)





