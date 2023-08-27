import pandas as pd

from model import *
from data_manager import DataManager

MODEL_TYPE = ModelType.LSTM

REG_PATH = './checkpoint/reg.pkl'
SAVE_STATE_DICT_PATH = './checkpoint/model.pth'

IS_REFRESH_VAR=0

# 获取数据
if IS_REFRESH_VAR or 'data_manager' not in vars():
    data_manager = DataManager(MODEL_TYPE)

# 获取模型
if IS_REFRESH_VAR or 'model' not in vars():
    model = Model(MODEL_TYPE)

# 获得线性回归模型
reg = Model(ModelType.LinearRegression)
X, y = data_manager.get_np_values()
reg.model.fit(X, y)
reg.save_model_as_pickle(REG_PATH)

# 训练模型
model.train(data_manager.get_training_data(), SAVE_STATE_DICT_PATH, epochs=5)

# 保存模型
model.save_model_as_state_dict(SAVE_STATE_DICT_PATH)

