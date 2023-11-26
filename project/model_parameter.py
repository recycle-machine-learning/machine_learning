import csv
import pandas as pd
import numpy as np
import torch


class model_parameter:
    def __init__(self):
        self.param = {}

    # 학습 후, 학습된 parameter를 이름과 함께 저장해 둠
    def save_parameter(self, param, name):
        np.set_printoptions(precision=7, suppress=True)
        param_np = param.detach().numpy()
        param_np = param_np.flatten()
        self.param[name] = param_np

    # 저장한 parameter를 하나로 합쳐 csv 파일로 저장
    def parameter_csv(self, epochs, learning_rate, train_batch_size,test_batch_size, size, out_channel1, out_channel2):
        hp = {"epochs": [epochs], "learning_rate": [learning_rate], "train_batch_size": [train_batch_size],
              "test_batch_size": [test_batch_size], "size": [size], "out_channel1": [out_channel1],
              "out_channel2": [out_channel2]}
        self.param.update(hp)
        df = pd.DataFrame.from_dict(self.param, orient= "index")
        df.to_csv("parameters.csv")


    def load_parameters(self,p_name, shape):
        df = pd.read_csv("parameters.csv")
        df = df.transpose()

        columns = df.loc['Unnamed: 0']
        df.rename(columns=columns, inplace = True)
        df = df.drop(['Unnamed: 0'],axis=0)

        parameter = df.loc[:, p_name]
        parameter = parameter.dropna(axis=0)
        parameter = parameter.to_frame()
        parameter = self.to_tensor(parameter,shape)
        return parameter

    def load_h_parameters(self,p_name):
        np.set_printoptions(precision=6, suppress=True)
        df = pd.read_csv("parameters.csv")
        df = df.transpose()

        columns = df.loc['Unnamed: 0']
        df.rename(columns=columns, inplace = True)
        df = df.drop(['Unnamed: 0'],axis=0)

        parameter = df.iloc[0][p_name]
        return parameter
    def to_tensor(self, p, shape):
        p = p.values
        p = p.reshape(shape)
        p = np.asarray(p, dtype = float)
        p_tensor = torch.from_numpy(p)
        p_tensor = p_tensor.type(torch.FloatTensor)
        return p_tensor

