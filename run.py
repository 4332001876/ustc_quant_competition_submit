import pandas as pd
import pickle
from rank_ic import *
import numpy as np

from model import *

MODEL_TYPE = ModelType.LinearNet

PICKLE_PATH = './checkpoint/reg.pkl'
SAVE_STATE_DICT_PATH = './checkpoint/pairwise linear/model13.pth'
# './checkpoint/pairwise linear/model10.pth'

#加载测试数据
test_df = pd.read_csv('../test.csv').set_index(['time_id', 'stock_id'])
test_df.fillna(0, inplace=True)
X_test = test_df.values

# 读取模型
'''
with open('reg.pkl', 'rb') as f:
    reg = pickle.load(f)'''
model = Model(MODEL_TYPE)
model.load_model_as_state_dict(SAVE_STATE_DICT_PATH)

#生成预测
if MODEL_TYPE==ModelType.LSTM:
    IS_GOURPBY_STOCK_ID = 1
    if IS_GOURPBY_STOCK_ID:
        result = None
        for index,data in test_df.iterrows():
            X = torch.tensor(data.values, dtype=torch.float32).reshape(-1,300)
            y_pred = model.predict((index[1],X)).reshape(-1)
            #temp_stock_df = pd.DataFrame(y_pred, index = group[1].index, columns=['pred'])
            if result is None:
                result = y_pred
            else:
                result = np.hstack((result, y_pred))
        result = pd.DataFrame(result, index = test_df.index, columns=['pred'])
    else:
        groups = test_df.groupby('stock_id')
        result = None
        for group in groups:
            group_result = None
            for i in range(0,len(group[1])):
                X = torch.tensor(group[1].iloc[0:i+1,:].values, dtype=torch.float32)
                y_pred = model.predict(X,dtype=torch.float32)
                if group_result is None:
                    group_result = y_pred
                else:
                    group_result = np.hstack((group_result, y_pred))
            temp_stock_df = pd.DataFrame(group_result, index = group[1].index, columns=['pred'])
            if result is None:
                result = temp_stock_df
            else:
                result = pd.concat([result, temp_stock_df], axis=0)
else:
    y_pred = model.predict(torch.tensor(X_test,dtype=torch.float32))
    result = pd.DataFrame(y_pred, index = test_df.index, columns=['pred'])

#保存结果
result = result.sort_index()
result.to_csv('./result.csv')


#计算rank_ic
parser = argparse.ArgumentParser()
parser.add_argument('--label_path', type=str,  default='../test_label.csv')
parser.add_argument('--result_path', type=str,  default='./result.csv')
args = parser.parse_args()

rank_ic = rank_ic(args.result_path, args.label_path)
print('rank_ic: ', rank_ic)




