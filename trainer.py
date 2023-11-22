import torch
import torch.optim as optim
import torch.nn as nn
from config import Config
from data_manager import DataManager
import pickle
from tqdm import tqdm

class Trainer:
    def __init__(self, model) -> None:
        self.model = model
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("Running on device:", self.device)
        self.model = self.model.to(self.device)
        if Config.PRETRAINED:
            self.load_model(Config.PRETRAINED_MODEL_PATH)

        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
        # self.optimizer = optim.SGD(model.parameters(), lr=Config.LEARNING_RATE, momentum=Config.MOMENTUM)
        """self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=Config.STEP_LR_STEP_SIZE,
            gamma=Config.STEP_LR_GAMMA,
            verbose=True,
        )"""
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = self.optimizer, T_max =  Config.EPOCHS+1, verbose = True)


        self.dataset_manager = DataManager(Config.MODEL_TYPE)
        self.train_loader = self.dataset_manager.get_training_data()
        self.test_loader = self.dataset_manager.get_training_data()
  
    def test_one_epoch(self, epoch):
        self.model.eval()
        running_loss = 0.0
        num_examples = 0
        batch_num = 0
        pbar = tqdm(total = len(self.test_loader), desc='Validating Epoch %d'%epoch)
        for batch in self.test_loader:
            inputs, targets = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            outputs = self.model(inputs)
            
            if outputs.shape != targets.shape:
                outputs = outputs.reshape(-1)
                targets = targets.reshape(-1)
            loss = self.loss_fn(outputs,targets) 
            running_loss += loss.data.item() # inputs.size(0)是batch_size
            batch_num += 1
            pbar.update(1)
        running_loss /= batch_num

        print('Epoch: {}, Validation Loss: {:.4f}'.format(epoch, running_loss))


    def train_one_epoch(self, epoch):
        running_loss = 0.0
        self.model.train()
        batch_num = 0
        pbar = tqdm(total = len(self.train_loader), desc='Training Epoch %d'%epoch)
        for batch in self.train_loader:
            self.optimizer.zero_grad()
            inputs, targets = batch    
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            outputs = self.model(inputs)

            if outputs.shape != targets.shape:
                outputs = outputs.reshape(-1)
                targets = targets.reshape(-1)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.data.item() # inputs.size(0)是batch_size
            batch_num += 1
            pbar.update(1)
        running_loss /= batch_num
        print('Epoch: {}, Training Loss: {:.4f}'.format(epoch, running_loss))


    def train(self):    
        for epoch in range(Config.EPOCHS):  # loop over the dataset multiple times
            self.train_one_epoch(epoch)
            self.scheduler.step()
            self.save_model(Config.SAVE_MODEL_PATH)
        self.save_model(Config.SAVE_FINAL_MODEL_PATH)
        print('Finished Training')

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def save_model_as_pickle(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load_model_as_pickle(self, path):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
