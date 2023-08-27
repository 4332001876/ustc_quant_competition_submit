import pandas as pd

from model import *
from data_manager import DataManager

PICKLE_PATH = './checkpoint/reg.pkl'
SAVE_STATE_DICT_PATH = './checkpoint/model.pth'

# 获取数据
data_manager = DataManager()

# 获取模型
model = Model()

# 训练模型
X, y = data_manager.get_np_values()
model.train(X, y)

# 保存模型
model.save_model_as_pickle(PICKLE_PATH)

