#我们想先得到所有的特征组合，然后分成n份存在列表里面
#启动n个程序来运算，可以节省时间
#这个文件中我们打算使用四个原子的五个属性加上最后一个原子的十个属性共计 30个特征来预测最终的结果GH
#同时我们也要进行参数搜索和五折交叉验证
#这里我们不再使用很蠢的方法,一个一个去试特征组合，而是使用RFE进行特征寻优
#RFE使用一个模型来拟合数据，并对每个特征赋予一个权重，然后，它会移除权重最小的特征，并用剩余的特征重新拟合模型。这个过程会不断重复，直到达到预设的特征数量。
from sklearn.preprocessing import scale,minmax_scale
import numpy as np
import joblib
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid, cross_val_score, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import make_scorer, r2_score, mean_squared_error
from tqdm import tqdm
import pandas as pd
import json
from math import sqrt
from sklearn.base import clone
from itertools import combinations
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, KFold,RandomizedSearchCV
import time
from joblib import dump
start_time = time.time()
#------------------------------------------------------读取数据集------------------------------------------------------#
# 假设你的数据集已经被加载到一个名为 data 的 pandas DataFrame 中
dataset=pd.read_excel('/scratch/uceezzz/Project/Baiproject/Dataset/G(H).xlsx')#这个是150条的
# dataset=dataset.loc[(dataset['G(H)']<24) & (dataset['G(H)']>-22)]
feature=dataset.iloc[:,1:-2] #总共是30个特征哈  注意，这个地方是取不到-2的，只能取到-3
label=dataset.iloc[:,-1].values

scaler = StandardScaler()
feature = scaler.fit_transform(feature)

#------------------------------------------------------计算特征组合------------------------------------------------------#
#计算一些排列组合，5个里面至少选三个
import itertools
features = list(range(20))
valid_combinations1 = []
valid_combinations2 = []
valid_combinations3 = []
valid_combinations4 = []
valid_combinations5 = []

#从前20个特征中的每个范围内选择3-5个特征
#但是我们后来发现，如果保证每个原子选择一样的特征，性能反而不是最好的，因此我们设置随机3到5个，不保证序号一样
# for i in range(3, 6):  # 选择的特征数量，从3到5, 16种组合,只修改这个地方来改变特征数量！！！！！！！！！
#     for indices in itertools.combinations(range(5), i):  # 在每个范围内选择特征的索引组合
#         combination = []
#         for start in range(0, 20, 5):  # 20：总的范围  5：每个原子的范围
#             combination.extend([start + idx for idx in indices])  # 添加符合要求的特征索引到组合中
#         valid_combinations1.append(combination)
# 在每个范围内选择3至5个特征
feature_min = 3
feature_max = 3
items = list(range(0,5))  # 为了减少计算难度，这里我们只使用了后7个特征
for i in range(feature_min, feature_max+1):  # 选择的特征个数范围
    combination = list(itertools.combinations(items, i))
    valid_combinations1.extend(combination)
print(len(valid_combinations1))
list_of_lists = []
for t in valid_combinations1:
    list_of_lists.append(list(t))
valid_combinations1 = list_of_lists


items = list(range(5,10))
for i in range(feature_min, feature_max+1):  # 选择的特征个数范围
    combination = list(itertools.combinations(items, i))
    valid_combinations2.extend(combination)
print(len(valid_combinations2))
list_of_lists = []
for t in valid_combinations2:
    list_of_lists.append(list(t))
valid_combinations2 = list_of_lists

items = list(range(10,15))
for i in range(feature_min, feature_max+1):  # 选择的特征个数范围
    combination = list(itertools.combinations(items, i))
    valid_combinations3.extend(combination)
print(len(valid_combinations3))
list_of_lists = []
for t in valid_combinations3:
    list_of_lists.append(list(t))
valid_combinations3 = list_of_lists

items = list(range(15,20))
for i in range(feature_min, feature_max+1):  # 选择的特征个数范围
    combination = list(itertools.combinations(items, i))
    valid_combinations4.extend(combination)
print(len(valid_combinations4))
list_of_lists = []
for t in valid_combinations4:
    list_of_lists.append(list(t))
valid_combinations4 = list_of_lists
#从后10个特征中的每个范围内选择3-7个特征
items = list(range(20,27))  # 为了减少计算难度，这里我们只使用了后7个特征
for i in range(4, 6):  # 选择的特征个数范围
    combination = list(itertools.combinations(items, i))
    valid_combinations5.extend(combination)
print(len(valid_combinations5))

list_of_lists = []
for t in valid_combinations5:
    list_of_lists.append(list(t))
valid_combinations5 = list_of_lists
#把两种组合合并,总共是
all_combinations = []

for comb1 in valid_combinations1:
    for comb2 in valid_combinations2:
        for comb3 in valid_combinations3:
            for comb4 in valid_combinations4:
                for comb5 in valid_combinations5:
                    all_combinations.append(comb1+comb2+comb3+comb4+comb5)

print(len(all_combinations))

#把这个列表分成十份
# 计算每个文件应包含的元素数量
size = len(all_combinations) // 10

# 将列表分成10份
for i in range(10):
    start_index = i * size
    end_index = (i + 1) * size if i < 9 else len(all_combinations)  # 对最后一份进行特殊处理以包含余数
    subset = all_combinations[start_index:end_index]
    dump(subset, f'file_{i+1}.joblib')