import pandas as pd

from model import *
from data_manager import DataManager

MODEL_TYPE = ModelType.LinearNet

PICKLE_PATH = './checkpoint/reg.pkl'
SAVE_STATE_DICT_PATH = './checkpoint/model.pth'

IS_REFRESH_VAR=0

# 获取数据
if IS_REFRESH_VAR or 'data_manager' not in vars():
    data_manager = DataManager(MODEL_TYPE)

# 获取模型
if IS_REFRESH_VAR or 'model' not in vars():
    model = Model(MODEL_TYPE)

# 训练模型
#X, y = data_manager.get_np_values()
model.train(data_manager.get_training_data(), SAVE_STATE_DICT_PATH, epochs=1)

# 保存模型
model.save_model_as_pickle(PICKLE_PATH)

