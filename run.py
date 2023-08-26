import pandas as pd
import pickle
from rank_ic import *

#加载测试数据
test_df = pd.read_csv('../test.csv').set_index(['time_id', 'stock_id'])
test_df.fillna(0, inplace=True)
X_test = test_df.values

# 读取模型
with open('reg.pkl', 'rb') as f:
    reg = pickle.load(f)

#生成预测
y_pred = reg.predict(X_test)

#保存结果
result = pd.DataFrame(y_pred, index = test_df.index, columns=['pred'])
result.to_csv('./result.csv')

#计算rank_ic
parser = argparse.ArgumentParser()
parser.add_argument('--label_path', type=str,  default='../test_label.csv')
parser.add_argument('--result_path', type=str,  default='./result.csv')
args = parser.parse_args()

rank_ic = rank_ic(args.result_path, args.label_path)
print('rank_ic: ', rank_ic)




