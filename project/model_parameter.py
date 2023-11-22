import os
import pandas as pd
import torch
import numpy as np

class model_parameter:
    def __init__(self):
        self.param = []
    def save_parameter(self,param):
        np.set_printoptions(precision=7, suppress=True)
        param_np = param.detach().numpy()
        self.param.append(param_np)

    def parameter_csv(self, name1, name2):
        weight_list = self.param[0]
        weight_list = weight_list.reshape(-1,3)
        bias_list = self.param[1]
        bias_list = bias_list.flatten()
        df1 = pd.DataFrame(weight_list)
        name1 = name1 + ".csv"
        name2 = name2 +".csv"
        df1.to_csv(name1,index = False)
        df2 = pd.DataFrame(bias_list)
        df2.to_csv(name2, index = False)
        return

