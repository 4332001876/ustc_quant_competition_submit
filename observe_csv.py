import pandas as pd

def gen_test_df():
    test_df = pd.DataFrame()
    test_df["time_id"] = [1, 1, 1, 2, 2, 2, 2]
    test_df["stock_id"] = [1, 2, 3, 1, 2, 3, 4]
    test_df["col1"] = [10, 12, 11, 16, 14, 15, 13]
    return test_df
test_df=gen_test_df()

def read_df(path):
    df = pd.read_csv(path)
    return df

class ObserveLines:
    def __init__(self):
        self.df=read_df('../test.csv')
        self.df=self.df.loc[0:5]
    def observe_one_line(self):
        line=self.df.iloc[0]
        print(line.to_list())
        return line

class GetMetaData:
    def __init__(self) -> None:
        self.df=read_df('../test.csv')
    def get_mean_and_std(self):
        col_mean = self.df.mean(axis=0)
        col_std = self.df.std(axis=0)
        return col_mean,col_std
    def get_data_count(self):
        count_df=self.df.groupby(['time_id','stock_id']).apply(lambda df: df.shape[0])
        print('stock_count:\n',count_df.groupby('time_id').apply(lambda df: df.shape[0]))
        print('time_count:\n',count_df.groupby('stock_id').apply(lambda df: df.shape[0]))
        

#observe_lines=ObserveLines()
#observe_lines.observe_one_line()

'''
get_meta_data=GetMetaData()
col_mean,col_std=get_meta_data.get_mean_and_std()
get_meta_data.get_data_count()
print('col_mean:\n',col_mean.to_list())
print('col_std:\n',col_std.to_list())
'''

#print(test_df.groupby('time_id').apply(lambda df: df['col1'].sum()))
#print(test_df['col1'].rank())

test_df=test_df.set_index(['time_id','stock_id'])
#test_df=test_df[test_df['stock_id']==1]
#test_df=test_df.set_index('stock_id').loc[1]#success
#print(test_df.loc[(1,2)])

'''for data in test_df:
    print(data)'''