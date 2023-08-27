import pandas as pd
class DataManager:
    def __init__(self):
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