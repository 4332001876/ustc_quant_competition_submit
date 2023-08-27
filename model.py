from sklearn import linear_model
import pickle
from enum import Enum

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

class Model:
    def __init__(self, model_type):
        self.model_type=model_type

        if self.model_type==ModelType.LinearRegression:
            self.model = linear_model.LinearRegression() # 创建线性回归模型
        elif self.model_type==ModelType.LinearNet:
            self.model = LinearNet()
        elif self.model_type==ModelType.LSTM:
            self.model = LSTM()

    def train(self, training_data, save_model_path, epochs=1):
        if self.model_type==ModelType.LinearRegression:
            features, labels = training_data
            self.model.fit(features, labels) # 线性回归模型
        elif self.model_type==ModelType.LinearNet:
            self.model.run_training(training_data, training_data, save_model_path, epochs=epochs)
        elif self.model_type==ModelType.LSTM:
            self.model.run_training(training_data, training_data, save_model_path, epochs=epochs)

    def predict(self, features):
        if self.model_type==ModelType.LinearRegression:
            return self.model.predict(features) # 线性回归模型
        elif self.model_type==ModelType.LinearNet:
            return self.model(features).detach().numpy()
        elif self.model_type==ModelType.LSTM:
            return self.model(features).detach().numpy()
    
    def save_model_as_pickle(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    def save_model_as_state_dict(self, path):
        torch.save(self.model.state_dict(), path)
    
    def load_model_as_pickle(self, path):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)

    def load_model_as_state_dict(self, path):
        self.model.load_state_dict(torch.load(path))

class LinearNet(nn.Module):
    def __init__(self):
        super(LinearNet, self).__init__()
        self.input_size = 300
        self.hidden_1_size = 18
        self.hidden_2_size = 6
        self.output_size = 1

        self.fc1 = nn.Linear(self.input_size, self.hidden_1_size)
        self.norm=torch.nn.BatchNorm1d(self.hidden_1_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.fc2 = nn.Linear(self.hidden_1_size, self.hidden_2_size)
        self.fc3 = nn.Linear(self.hidden_2_size, self.output_size)

        self.reg = Model(ModelType.LinearRegression)
        self.reg.load_model_as_pickle(REG_BASELINE_PATH)
    
    def forward(self, x):
        x = x.view(-1, self.input_size)
        reg_result = self.reg.predict(x.detach().numpy())
        x = F.relu(self.fc1(x))
        x = self.norm(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x + torch.tensor(reg_result, dtype=torch.float32).reshape(-1, 1)
        average = torch.mean(x)
        x = x - average
        return x
    
    def run_training(self, train_loader, val_loader, save_model_path, epochs=5):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        loss_fn = nn.MSELoss()
        train(self, optimizer, loss_fn, train_loader, val_loader, save_model_path, epochs=epochs)

class LSTM(nn.Module):
    def __init__(self, input_size=300, hidden_layer_size=20, output_size=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell_pool={}

        self.reg = Model(ModelType.LinearRegression)
        self.reg.load_model_as_pickle(REG_BASELINE_PATH)
        
    def forward(self, inputs):
        stock_id, input_seq = inputs
        if stock_id not in self.hidden_cell_pool:
            self.hidden_cell_pool[stock_id] = (torch.randn(1,1,self.hidden_layer_size), torch.randn(1,1,self.hidden_layer_size))
            self.hidden_cell = self.hidden_cell_pool[stock_id]
        else:
            self.hidden_cell = self.hidden_cell_pool[stock_id]

        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        result = predictions.reshape(-1, 1)
        reg_result = self.reg.predict(input_seq.detach().numpy().reshape(-1,self.input_size))
        result += torch.tensor(reg_result, dtype=torch.float32).reshape(-1, 1)

        return result
    
    def run_training(self, train_loader, val_loader, save_model_path, epochs=1):
        optimizer = optim.Adam(self.parameters(), lr=0.0001)
        loss_fn = nn.MSELoss()
        train(self, optimizer, loss_fn, train_loader, val_loader, save_model_path, epochs=epochs)
    
def train(model, optimizer, loss_fn, train_loader, val_loader, save_model_path, epochs=1, need_correct_rate = False):
    IS_LSTM=1
    if torch.cuda.is_available():
        device = torch.device("cuda") 
    else:
        device = torch.device("cpu")
        
    for epoch in range(1, epochs+1):
        training_loss = 0.0
        valid_loss = 0.0

        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, targets = batch
            #inputs = inputs.to(device)
            #targets = targets.to(device)
            output = model(inputs)
            if output.shape != targets.shape:
                output = output.reshape(-1)
                targets = targets.reshape(-1)
            loss = loss_fn(output, targets)
            loss.backward()
            optimizer.step()
            training_loss += loss.data.item() # inputs.size(0)是batch_size
        training_loss /= len(train_loader)
        
        model.eval()
        num_correct = 0 
        num_examples = 0
        for batch in val_loader:
            inputs, targets = batch
            #inputs = inputs.to(device)
            output = model(inputs)
            #targets = targets.to(device)
            if output.shape != targets.shape:
                output = output.reshape(-1)
                targets = targets.reshape(-1)
            loss = loss_fn(output,targets) 
            valid_loss += loss.data.item() # inputs.size(0)是batch_size
            if need_correct_rate: # 结果是个size不为1的向量
                correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], targets)
            else:
                correct = torch.tensor([0])
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]
        valid_loss /= len(val_loader)

        print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}, accuracy = {:.4f}'.format(epoch, training_loss,
        valid_loss, num_correct / num_examples))
        torch.save(model.state_dict(),save_model_path)