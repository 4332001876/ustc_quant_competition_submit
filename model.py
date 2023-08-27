from sklearn import linear_model
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F

class Model:
    def __init__(self):
        self.model = linear_model.LinearRegression() # 创建线性回归模型

    def train(self, features, labels, epochs=5):
        self.model.fit(features, labels) # 训练模型

    def predict(self, features):
        return self.model.predict(features)
    
    def save_model_as_pickle(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)


class LinearNet(nn.Module):
    def __init__(self):
        super(LinearNet, self).__init__()
        self.input_size = 300
        self.hidden_1_size = 30
        self.hidden_2_size = 12
        self.output_size = 1

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
        return x
    def run_training(self, train_loader, val_loader, save_model_path, epochs=1):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()
        train(self, optimizer, loss_fn, train_loader, val_loader, save_model_path, epochs=epochs)

class LSTM(nn.Module):
    def __init__(self):
        super(LinearNet, self).__init__()
        self.input_size = 300
        self.hidden_1_size = 30
        self.hidden_2_size = 12
        self.output_size = 1

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
        return x
    def run_training(self, train_loader, val_loader, save_model_path, epochs=1):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()
        train(self, optimizer, loss_fn, train_loader, val_loader, save_model_path, epochs=epochs)
    
def train(model, optimizer, loss_fn, train_loader, val_loader, save_model_path, epochs=1):
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
            loss = loss_fn(output, targets)
            loss.backward()
            optimizer.step()
            training_loss += loss.data.item() * inputs.size(0) # inputs.size(0)是batch_size
        training_loss /= len(train_loader.dataset)
        
        model.eval()
        num_correct = 0 
        num_examples = 0
        for batch in val_loader:
            inputs, targets = batch
            #inputs = inputs.to(device)
            output = model(inputs)
            #targets = targets.to(device)
            loss = loss_fn(output,targets) 
            valid_loss += loss.data.item() * inputs.size(0) # inputs.size(0)是batch_size
            if targets[0].size()!=torch.Size([]): # 结果是个size不为1的向量
                correct=torch.tensor([0])
            else:
                correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], targets)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]
        valid_loss /= len(val_loader.dataset)

        print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}, accuracy = {:.4f}'.format(epoch, training_loss,
        valid_loss, num_correct / num_examples))
        torch.save(model.state_dict(),save_model_path)