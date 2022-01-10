
# ### 纲要
# - 将数据统一到data2类型 connect
# - Mean std取data2的
# - Train val test 比例

import pandas as pd
import warnings
import pickle
import datetime
warnings.filterwarnings('ignore')
from torch.utils.data import DataLoader,Dataset
import numpy as np
import torch


# ### 数据项读取以及预处理
# - 先将 train_public.csv 另存为 train_public2.csv，并对earlies_credit_mon改成短日期格式 ！！
# - test_public 同上

# %%
train_pub = pd.read_csv('./train_public2.csv')
test_pub = pd.read_csv('./test_public2.csv')
train_internet = pd.read_csv('./train_internet.csv')

# %%
print("##Public index:")
print(train_pub.columns.intersection(train_internet.columns))
print("##Only in train_pub:")
print(train_pub.columns.difference(train_internet.columns))
print("##Only in train_internet:")
print(train_internet.columns.difference(train_pub.columns))

# %%
# 将短日期格式的 2021/12/1 => 2001-12-01 （这里2021应该是系统自动添加上的，实际为 12/1，即月/年）
def format_date(x):
    if x>= pd.to_datetime('2021-01-01'):
        t = '20' + str(x)[8:10] + '-' + str(x)[5:7] + '-01'
        #print('t=', t)
        return pd.to_datetime(t)
    return x

issue_date_base_time = datetime.datetime.strptime('2007-06-01', '%Y-%m-%d')
earlies_credit_base_time = datetime.datetime.strptime('1952-06-01', '%Y-%m-%d')
# internet_issue_date_base_time = 

employer_type = train_internet['employer_type'].value_counts().index
industry = train_internet['industry'].value_counts().index
work_year = train_internet['work_year'].value_counts().index
class_index = train_internet['class'].value_counts().index


# 标签编码
emp_type_dict = dict(zip(employer_type, [i for i in range(len(employer_type))]))
industry_dict = dict(zip(industry, [i for i in range(len(industry))]))
work_year_dict = dict(zip(work_year, [i for i in range(len(work_year))]))
class_dict = dict(zip(class_index, [i for i in range(len(class_index))]))

################### train public format ####################
train_pub['earlies_credit_mon'] = pd.to_datetime(train_pub['earlies_credit_mon'])
train_pub['earlies_credit_mon'] = train_pub['earlies_credit_mon'].apply(format_date)
train_pub['issue_date'] = pd.to_datetime(train_pub['issue_date'])
train_pub['issue_date_diff'] = train_pub['issue_date'].apply(lambda x: x - issue_date_base_time).dt.days
train_pub['earlies_credit_mon_diff'] = train_pub['earlies_credit_mon'].apply(lambda x: x - earlies_credit_base_time).dt.days
train_pub['issue_earlies_diff'] = (train_pub['issue_date'] - train_pub['earlies_credit_mon']).dt.days
train_pub.drop('issue_date', axis = 1, inplace = True)


################### test public format ####################
test_pub['earlies_credit_mon'] = pd.to_datetime(test_pub['earlies_credit_mon'])
test_pub['earlies_credit_mon'] = test_pub['earlies_credit_mon'].apply(format_date)
test_pub['issue_date'] = pd.to_datetime(test_pub['issue_date'])
test_pub['issue_date_diff'] = test_pub['issue_date'].apply(lambda x: x - issue_date_base_time).dt.days
test_pub['earlies_credit_mon_diff'] = test_pub['earlies_credit_mon'].apply(lambda x: x - earlies_credit_base_time).dt.days
test_pub['issue_earlies_diff'] = (test_pub['issue_date'] - test_pub['earlies_credit_mon']).dt.days
test_pub.drop('issue_date', axis = 1, inplace = True)

################### train internet format ####################
train_internet['earlies_credit_mon'] = pd.to_datetime(train_internet['earlies_credit_mon'])
train_internet['earlies_credit_mon'] = train_internet['earlies_credit_mon'].apply(format_date)
# 因为数据集的特殊性 不可以用标准数据集的base time  需要另行计算
internet_earlies_credit_base_time = min(train_internet['earlies_credit_mon'])
internet_issue_date_base_time = min(train_internet['issue_date'])

train_internet['issue_date'] = pd.to_datetime(train_internet['issue_date'])
train_internet['issue_date_diff'] = train_internet['issue_date'].apply(lambda x: x - internet_earlies_credit_base_time).dt.days
train_internet['earlies_credit_mon_diff'] = train_internet['earlies_credit_mon'].apply(lambda x: x - internet_earlies_credit_base_time).dt.days
train_internet['issue_earlies_diff'] = (train_internet['issue_date'] - train_internet['earlies_credit_mon']).dt.days
train_internet.drop('issue_date', axis = 1, inplace = True)


# %%
print(train_pub.shape)
print(test_pub.shape)
print(train_internet.shape)

# %%
common_index = test_pub.columns[2:].intersection(train_internet.columns)
only_in_pub_index = test_pub.columns[2:].difference(train_internet.columns)
# loan_id  user_id 不用于训练
concat_index = test_pub.columns[2:]
all_features = pd.concat((train_pub[concat_index], test_pub[concat_index], train_internet[common_index]))

all_features['class'] = all_features['class'].map(class_dict)
all_features['employer_type'] = all_features['employer_type'].map(emp_type_dict)
all_features['industry'] = all_features['industry'].map(industry_dict)
all_features['work_year']  = all_features['work_year'].map(work_year_dict)

print(all_features.shape)


# %%
n_train = train_pub.shape[0]
n_test = test_pub.shape[0]
n_inter_train = train_internet.shape[0]

pub_features = all_features[: -n_inter_train]
inter_train_features = all_features[-n_inter_train: ]


# %%
inter_train_features.duplicated()

# %%
inter_train_features = inter_train_features.apply(
    lambda x: (x - x.mean()) / (x.std()))

# 标准化后，每个数值特征的均值变为0，所以可以直接用0来替换缺失值
inter_train_features = inter_train_features.fillna(0)

pub_features = pub_features.apply(
    lambda x: (x - x.mean()) / (x.std()))
pub_features = pub_features.fillna(0)

train_features = pub_features[: n_train]
test_features = pub_features[n_train: ]

# %%
# 去重
# df.drop_duplicates(inplace=True)


class MyDataset(Dataset):

    def __init__(self):
        self.test_data = torch.tensor(np.array(train_features))
        self.len = n_train

    def __getitem__(self, index):
        return self.test_data[index]

    def __len__(self):
        return self.len



