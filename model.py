from sklearn import linear_model
import pickle
from enum import Enum
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from torchmetrics.regression import MeanAbsolutePercentageError

REG_BASELINE_PATH = './checkpoint/reg baseline.pkl'

class ModelType(Enum):
    LinearRegression=1
    LinearNet=2
    LSTM=3

def PairWiseLoss(outputs, targets, margin=0.1):
    loss = torch.tensor(0, dtype=torch.float32)
    loss.requires_grad_(True)
    idx = [i for i in range(len(outputs))]
    random.shuffle(idx)
    for n in range(len(outputs)//2):
        i = idx[2*n]
        j = idx[2*n+1]
        if targets[i] > targets[j]:
            if margin-outputs[i]+outputs[j]>0:
                loss = loss + margin-outputs[i]+outputs[j]
        else:
            if margin-outputs[j]+outputs[i]>0:
                loss = loss + margin-outputs[j]+outputs[i]
        loss = loss
    return loss
    
class LinearRegression(nn.Module):
    def __init__(self):
        self.model = linear_model.LinearRegression() # 创建线性回归模型

    def train(self, training_data):
        features, labels = training_data
        self.model.fit(features, labels) # 线性回归模型

    def forward(self, features):
        return self.model.predict(features) # 线性回归模型

class LinearNet(nn.Module):
    def __init__(self):
        super(LinearNet, self).__init__()
        self.input_size = 300
        self.hidden_1_size = 180
        self.hidden_2_size = 60
        self.output_size = 10

        self.fc1 = nn.Linear(self.input_size, self.hidden_1_size)
        self.norm=torch.nn.BatchNorm1d(self.hidden_1_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.fc2 = nn.Linear(self.hidden_1_size, self.hidden_2_size)
        self.fc3 = nn.Linear(self.hidden_2_size, self.output_size)
    
    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = F.relu(self.fc1(x))
        x = self.norm(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.mean(x, dim=1)
        return x

class LSTM(nn.Module):
    def __init__(self, input_size=300, hidden_layer_size=60, output_size=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell_pool={}

        self.linear_net = LinearRegression()
        self.linear_net.load_model_as_state_dict('./checkpoint/pairwise linear/model15.pth')
        self.linear_net.model.eval()
        
    def forward(self, inputs):
        stock_id, input_seq = inputs
        if stock_id not in self.hidden_cell_pool:
            self.hidden_cell_pool[stock_id] = (torch.randn(1,1,self.hidden_layer_size), torch.randn(1,1,self.hidden_layer_size))
            self.hidden_cell = self.hidden_cell_pool[stock_id]
        else:
            self.hidden_cell = self.hidden_cell_pool[stock_id]

        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        result = predictions[-1].reshape(-1, 1)
        linear_result = self.linear_net.predict(input_seq[-1].reshape(-1,self.input_size))
        linear_weight = torch.tensor(1, dtype=torch.float32, requires_grad=True)
        result += linear_weight * torch.tensor(linear_result, dtype=torch.float32).reshape(-1, 1)

        return result
    
