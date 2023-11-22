import pandas as pd
import argparse
from rank_ic import *

from model import *
from trainer import Trainer
from config import Config

model = LinearNet()
trainer = Trainer(model)
trainer.train(Config.EPOCHS)

test_df = pd.read_csv(Config.TEST_DATASET_PATH).set_index(['time_id', 'stock_id'])
test_df.fillna(0, inplace=True)
X_test = test_df.values
y_pred = model(torch.tensor(X_test,dtype=torch.float32))
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


