import pandas as pd
from sklearn import linear_model
import pickle

from config import Config
from rank_ic import *


# 读取数据, 选择 'time_id', 'stock_id' 作为索引
train_df = pd.read_csv(Config.DATASET_PATH).set_index(['time_id', 'stock_id'])

# 填充缺失数据
train_df.fillna(0, inplace=True)
print(train_df)

# 创建线性回归模型
reg = linear_model.LinearRegression()

# 将数据集划分为特征和目标变量
X = train_df.iloc[:,:-1].values
y = train_df.iloc[:,-1].values

# 训练模型
reg.fit(X, y)

# 保存模型
with open(Config.REG_PATH, 'wb') as f:
    pickle.dump(reg, f)

#加载测试数据
test_df = pd.read_csv(Config.TEST_DATASET_PATH).set_index(['time_id', 'stock_id'])
test_df.fillna(0, inplace=True)
X_test = test_df.values

#生成预测
y_pred = reg.predict(X_test)

#保存结果
result = pd.DataFrame(y_pred, index = test_df.index, columns=['pred'])
result.to_csv(Config.SAVE_RESULT_PATH)

#计算rank_ic
rank_ic = rank_ic(Config.SAVE_RESULT_PATH, Config.TEST_DATASET_LABEL_PATH) 
print('rank_ic: ', rank_ic)