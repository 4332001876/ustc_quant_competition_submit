import pandas as pd
from model import ModelType

import torch
import torch.utils.data

class DataManager:
    def __init__(self, model_type):
        self.model_type = model_type
        # 读取数据, 选择 'time_id', 'stock_id' 作为索引
        self.train_df = pd.read_csv('../train.csv').set_index(['time_id', 'stock_id'])
        # 填充缺失数据
        self.train_df.fillna(0, inplace=True)
        print(self.train_df)
    def get_np_values(self):
        # 将数据集划分为特征和目标变量
        X = self.train_df.iloc[:,:-1].values
        y = self.train_df.iloc[:,-1].values
        return X,y
    def get_time_id_grouped_data(self):
        groups = self.train_df.groupby('time_id')
        time_id_grouped_data=[]
        for group in groups:
            X = torch.tensor(group[1].iloc[:,:-1].values, dtype=torch.float32)
            y = torch.tensor(group[1].iloc[:,-1].values, dtype=torch.float32)
            time_id_grouped_data.append((X, y))
        return time_id_grouped_data
    
    def get_time_id_grouped_df(self):
        groups = self.train_df.groupby('time_id')
        time_id_grouped_df=[]
        for group in groups:
            time_id_grouped_df.append(group)
        return time_id_grouped_df
    
    def get_stock_id_grouped_data(self):
        groups = self.train_df.groupby('stock_id')
        stock_id_grouped_data=[]
        for group in groups:
            X = torch.tensor(group[1].iloc[:,:-1].values, dtype=torch.float32)
            y = torch.tensor(group[1].iloc[:,-1].values, dtype=torch.float32)
            stock_id_grouped_data.append(((group[0],X), y))
        return stock_id_grouped_data

    def get_training_data(self):
        X,y = self.get_np_values()
        if self.model_type==ModelType.LinearRegression:
            return (X,y)
        elif self.model_type==ModelType.LinearNet:
            #return torch.utils.data.DataLoader(Dataset(X,y), batch_size=64, shuffle=True)
            return self.get_time_id_grouped_data()
        elif self.model_type==ModelType.LSTM:
            return self.get_time_id_grouped_df()
        
class Dataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def __getitem__(self, index):    
        return torch.tensor(self.features[index],dtype=torch.float32), torch.tensor(self.labels[index],dtype=torch.float32)
    
    def __len__(self):
        return len(self.features)
    
