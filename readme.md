## 项目结构
模型训练：
python train.py

模型预测：
python run.py

结果评估:
python rank_ic.py 
(参数 --label_path, --result_path)

## 数据集结构
### 训练集
文件名：train.csv

格式：`time_id,stock_id,feature0,feature1,...,feature299,label`

时间点0-727共728个时间点，每一时刻约有1700-1800支股票，共2338支股票出现过，数据行数为1284587行


### 测试集
文件名：test.csv

格式：`time_id,stock_id,feature0,feature1,...,feature299`

时间点747-844共98个时间点，每一时刻约有1700-1800支股票，共1908支股票出现过，数据行数为--行


文件名：test_label.csv

格式：`time_id,stock_id,label`

## 测试结果
baseline-线性回归：0.0926
naive linear net: 0.0301



