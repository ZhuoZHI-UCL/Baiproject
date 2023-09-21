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
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.svm import SVC  # 改为SVC用于分类
from sklearn.neural_network import MLPClassifier 
start_time = time.time()
#------------------------------------------------------读取数据集------------------------------------------------------#
# 假设你的数据集已经被加载到一个名为 data 的 pandas DataFrame 中
dataset=pd.read_excel('/scratch/uceezzz/Project/Baiproject/Dataset/0920/训练-非平均.xlsx')#这个是150条的
# dataset=dataset.loc[(dataset['G(H)']<24) & (dataset['G(H)']>-22)]
feature=dataset.iloc[:,1:-3] #总共是27个特征哈 注意右边是取不到的，只能取到前面一个哈
label=dataset.iloc[:,-1].values

scaler = StandardScaler()
feature = scaler.fit_transform(feature)

#------------------------------------------------------计算特征组合------------------------------------------------------#
#计算一些排列组合，5个里面至少选三个
import itertools
valid_combinations = []
items = list(range(len(feature[0])))  # 为了减少计算难度，这里我们只使用了后7个特征
for i in range(3, len(feature[0])):  # 选择的特征个数范围
    combination = list(itertools.combinations(items, i))
    valid_combinations.extend(combination)

#------------------------------------------------------参数寻优------------------------------------------------------#

# 指定你要使用的回归器及其参数网格
classifiers = {
    "SVC": (SVC(), {
        "kernel": ["linear", "rbf"],
        "C": [0.1, 1, 10],
        "gamma": ["scale", "auto", 0.1, 1],
    }),
    "MLPClassifier": (MLPClassifier(), {
        "hidden_layer_sizes": [(50,), (100,)],
        "activation": ["relu", "tanh"],
        "learning_rate": ["constant", "invscaling"],
        "max_iter": [200, 300, 500],
        "alpha": [ 0.001, 0.01],
    }),
}

all_feature_combinations = valid_combinations

kf = KFold(n_splits=5, shuffle=True, random_state=0)

best_f1_score = -1e9
best_rmse = 1e9
best_classifier  = None
best_params = None
best_feature_combination = None
total_combinations = len(all_feature_combinations)

with open('/scratch/uceezzz/Project/Baiproject/Code/09.20/非平均/Best_pre_Ebin.txt', 'w') as file:
    for idx, feature_combination in tqdm(enumerate(all_feature_combinations), total=len(all_feature_combinations)):
        selected_features = feature[:, feature_combination]
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"已消耗的时间: {elapsed_time:.2f} 秒")
        for name, (classifier, params) in classifiers.items():

            f1_scorer = make_scorer(f1_score) 

            grid_search = GridSearchCV(classifier, params, scoring={'f1': f1_scorer}, cv=kf, n_jobs=-1,refit='f1')
            
            grid_search.fit(selected_features, label)

            best_estimator = grid_search.best_estimator_
            best_params_combination = grid_search.best_params_
            # 获取cv_results
            cv_results = grid_search.cv_results_
            
            mean_f1 = cv_results['mean_test_f1'][grid_search.best_index_]  # 改为使用f1作为评价指标
            
            # 计算r2和rmse的标准差
            std_f1 = cv_results['std_test_f1'][grid_search.best_index_]

            # 判断是否是最佳组合
            if mean_f1  > best_f1_score :
                best_f1_score = mean_f1
                best_classifier = name  
                best_params = best_params_combination
                best_feature_combination = feature_combination
                best_std_f1 = std_f1
                    
                file.seek(0)
                file.truncate()
                
                file.write("最佳f1分数: {}\n".format(best_f1_score))
                file.write("最佳特征组合: {}\n".format(best_feature_combination))
                file.write("最佳模型: {}\n".format(best_classifier))
                file.write("最佳参数: {}\n".format(best_params))
                file.write("f1方差: {}\n".format(best_std_f1))  
                file.flush()





