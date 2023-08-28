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
            loss += torch.max(torch.tensor([0.0,margin-outputs[i]+outputs[j]], dtype=torch.float32))
        else:
            loss += torch.max(torch.tensor([0.0,margin-outputs[j]+outputs[i]], dtype=torch.float32))
    return loss

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
        loss_fn = PairWiseLoss # nn.MSELoss()
        train(self, optimizer, loss_fn, train_loader, val_loader, save_model_path, epochs=epochs)

class LSTM(nn.Module):
    def __init__(self, input_size=300, hidden_layer_size=60, output_size=1):
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
        result = predictions[-1].reshape(-1, 1)
        reg_result = self.reg.predict(input_seq[-1].detach().numpy().reshape(-1,self.input_size))
        result += torch.tensor(reg_result, dtype=torch.float32).reshape(-1, 1)

        return result
    
    def run_training(self, train_loader, val_loader, save_model_path, epochs=1):
        optimizer = optim.Adam(self.parameters(), lr=0.0001)
        loss_fn = PairWiseLoss# nn.MSELoss()
        train(self, optimizer, loss_fn, train_loader, val_loader, save_model_path, epochs=epochs)
    
def train(model, optimizer, loss_fn, train_loader, val_loader, save_model_path, epochs=1, need_correct_rate = False):
    IS_LSTM=0
    if torch.cuda.is_available():
        device = torch.device("cuda") 
    else:
        device = torch.device("cpu")
        
    for epoch in range(1, epochs+1):
        training_loss = 0.0
        valid_loss = 0.0

        model.train()
        batch_num = 0
        pbar = tqdm(total = len(train_loader), desc='Training Epoch %d'%epoch)
        for batch in train_loader:
            optimizer.zero_grad()
            

            if IS_LSTM:
                outputs = None
                for index, data in batch[1].iterrows():
                    X = torch.tensor(data[:-1].values, dtype=torch.float32).reshape(-1,300)
                    input = (index[1],X)
                    output = model(input)
                    if outputs is None:
                        outputs = output
                    else:
                        outputs = torch.cat((outputs, output), 0)
                targets = torch.tensor(batch[1].iloc[:,-1].values, dtype=torch.float32).reshape(-1,1)
                outputs = outputs - torch.mean(outputs)
            else:
                inputs, targets = batch    
                #inputs = inputs.to(device)
                #targets = targets.to(device)
                outputs = model(inputs)

            if outputs.shape != targets.shape:
                outputs = outputs.reshape(-1)
                targets = targets.reshape(-1)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            training_loss += loss.data.item() # inputs.size(0)是batch_size
            batch_num += 1
            pbar.update(1)
        training_loss /= batch_num
        
        '''model.eval()
        num_correct = 0 
        num_examples = 0
        batch_num = 0
        pbar = tqdm(total = len(train_loader), desc='Validating Epoch %d'%epoch)
        for batch in val_loader:
            inputs, targets = batch
            #inputs = inputs.to(device)
            if IS_LSTM:
                rand_index = random.randint(0,targets.size(0)-1)
                inputs = (inputs[0], inputs[1][rand_index].reshape(1,-1))
                targets = targets[rand_index]

            outputs = model(inputs)
            #targets = targets.to(device)
            if outputs.shape != targets.shape:
                outputs = outputs.reshape(-1)
                targets = targets.reshape(-1)
            loss = loss_fn(outputs,targets) 
            valid_loss += loss.data.item() # inputs.size(0)是batch_size
            if need_correct_rate: # 结果是个size不为1的向量
                correct = torch.eq(torch.max(F.softmax(outputs, dim=1), dim=1)[1], targets)
            else:
                correct = torch.tensor([0])
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]
            batch_num += 1
            pbar.update(1)
        valid_loss /= batch_num

        print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}, accuracy = {:.4f}'.format(epoch, training_loss,
        valid_loss, num_correct / num_examples))'''
        print('Epoch: {}, Training Loss: {:.4f}'.format(epoch, training_loss))
        torch.save(model.state_dict(),save_model_path)
        if epoch%1==0:
            torch.save(model.state_dict(),'./checkpoint/model%d.pth'%epoch)


