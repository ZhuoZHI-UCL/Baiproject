#预测GH
from sklearn.preprocessing import scale,minmax_scale
from sklearn.metrics import make_scorer, mean_squared_error
import numpy as np
import joblib
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid, cross_val_score, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import make_scorer, r2_score, mean_squared_error, f1_score
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
start_time = time.time()
#------------------------------------------------------读取数据集------------------------------------------------------#
# 假设你的数据集已经被加载到一个名为 data 的 pandas DataFrame 中
dataset=pd.read_excel('/scratch/uceezzz/Project/Baiproject/Dataset/0814_EF.xlsx')#这个是150条的
test_dataset = pd.read_excel('/scratch/uceezzz/Project/Baiproject/Dataset/prediction_source.xlsx')#这个是50条的

feature=dataset.iloc[:,1:-2] #总共是27个特征哈  注意，这个地方是取不到-2的，只能取到-3
feature = feature.iloc[:,[0, 2, 3, 5, 7, 8, 10, 12, 13, 15, 17, 18, 20, 21, 25, 26]]
label=dataset.iloc[:,-2].values

test_feature = test_dataset.iloc[:,1:]
test_feature = test_feature.iloc[:,[0, 2, 3, 5, 7, 8, 10, 12, 13, 15, 17, 18, 20, 21, 25, 26]]


scaler = StandardScaler()
feature = scaler.fit_transform(feature)
test_feature = scaler.fit_transform(test_feature)
#------------------------------------------------------参数寻优------------------------------------------------------#

# 指定你要使用的回归器及其参数网格
# 定义MLP模型
model = MLPClassifier(hidden_layer_sizes=50, 
                     activation='relu', 
                     learning_rate='constant', 
                     max_iter=200,
                     random_state=1,
                     alpha = 0.01)

# 定义交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=0)

# 用于存储得分的列表
f1_scores = []
test_pred_Kfold = []

# 进行五折交叉验证
for train_index, test_index in kf.split(feature):
    X_train, X_test = feature[train_index], feature[test_index]
    y_train, y_test = label[train_index], label[test_index]

    # 训练模型
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    y = np.array(y_test)
    y_pred = np.array(y_pred)
    diff = abs(y - y_pred)
    # num=6#100个点
    num = 0
    max_index = np.argpartition(-diff, num)[:num]
    y_pred = np.delete(y_pred, max_index)
    y = np.delete(y, max_index)

    
    
    
    # 计算得分
    f1_scores.append(f1_score(y, y_pred))
    
    #预测test的数据
    test_pred = model.predict(test_feature)
    test_pred_Kfold.append(test_pred)
    
test_pred_mean = np.mean(np.array(test_pred_Kfold),axis=0)
# 将小于0.5的数变成0，大于0.5的数变成1
test_pred_mean[test_pred_mean < 0.5] = 0
test_pred_mean[test_pred_mean > 0.5] = 1
#替换极端的m个值,把m个最大的值替换成第m个最大的值，把m个最小的值替换成第m个最小的值
# m=int(0.02*len(test_pred_mean))
# max_indices = test_pred_mean.argsort()[-m:]
# min_indices = test_pred_mean.argsort()[:m]
# replacement_value_max = np.partition(test_pred_mean, -m)[-m]
# replacement_value_min = np.partition(test_pred_mean, m-1)[m-1]
# test_pred_mean[max_indices] = replacement_value_max
# test_pred_mean[min_indices] = replacement_value_min

# 计算并打印得分的平均值和标准差
print('f1 mean: ', np.mean(f1_scores))
print('f1 std: ', np.std(f1_scores))

#把预测的值插入到test_dataset中
prediction_dataset = pd.read_excel('/scratch/uceezzz/Project/Baiproject/Dataset/prediction.xlsx')#这个是50条的
prediction_dataset['EF'] = test_pred_mean
prediction_dataset.to_excel('/scratch/uceezzz/Project/Baiproject/Dataset/prediction.xlsx')
print('All done!')

'''    
f1 mean:  0.9111604641856743
f1 std:  0.020758924032625668
'''     





