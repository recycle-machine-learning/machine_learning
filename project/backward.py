import torch
from collections import OrderedDict
from layers import *
class Backward:

    def __init__(self, model):
        self.model = model
        self.relu1 = model.relu1
        self.relu2 = model.relu2
        self.conv1 = model.conv1_test
        self.bn1 = model.bn1
        self.pool1 = model.pool1_test
        self.conv2 = model.conv2_test
        self.bn2 = model.bn2
        self.pool2 = model.pool2_test
        self.fc1 = model.fc1_test
        # self.drop1 = model.drop1
        # self.drop2 = model.drop2
        # self.drop3 = model.drop3


    def backward(self, x):
        # dout = self.drop3.backward(x)
        dout = self.fc1.backward(x)

        dout = self.pool2.backward(dout)
        # dout = self.drop2.backward(dout)
        dout = self.relu2.backward(dout)
        # dout = self.bn2.backward(dout)
        dout = self.conv2.backward(dout)

        dout = self.pool1.backward(dout)
        # dout = self.drop1.backward()
        dout = self.relu1.backward(dout)

        # dout = self.bn1.backward()
        dout = self.conv1.backward(dout)





