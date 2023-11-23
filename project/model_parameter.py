import os
import pandas as pd
import torch
import numpy as np


class model_parameter:
    def __init__(self):
        self.param = {}

    def save_parameter(self, param, name):
        np.set_printoptions(precision=7, suppress=True)
        param_np = param.detach().numpy()
        param_np = param_np.flatten()
        self.param[name] = param_np

    def parameter_csv(self, epochs, learning_rate, train_batch_size,test_batch_size, size, out_channel1, out_channel2):
        hp = {"epochs": [epochs], "learning_rate": [learning_rate], "train_batch_size": [train_batch_size],
              "test_batch_size": [test_batch_size], "size": [size], "out_channel1": [out_channel1],
              "out_channel2": [out_channel2]}
        self.param.update(hp)
        df = pd.DataFrame.from_dict(self.param, orient= "index")
        df.to_csv("parameters.csv")
