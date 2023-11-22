import pandas as pd
import argparse
from rank_ic import *

from model import *
from trainer import Trainer
from config import Config

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = LinearNet()
trainer = Trainer(model)
trainer.train()

test_df = pd.read_csv(Config.TEST_DATASET_PATH).set_index(['time_id', 'stock_id'])
test_df.fillna(0, inplace=True)
X_test = test_df.values
y_pred = model(torch.tensor(X_test,dtype=torch.float32).to(device)).cpu().detach().numpy()
result = pd.DataFrame(y_pred, index = test_df.index, columns=['pred'])

#保存结果
result = result.sort_index()
result.to_csv(Config.SAVE_RESULT_PATH)

#计算rank_ic
parser = argparse.ArgumentParser()
parser.add_argument('--label_path', type=str,  default=Config.TEST_DATASET_LABEL_PATH)
parser.add_argument('--result_path', type=str,  default=Config.SAVE_RESULT_PATH)
args = parser.parse_args()

rank_ic = rank_ic(args.result_path, args.label_path)
print('rank_ic: ', rank_ic)


