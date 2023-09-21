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
start_time = time.time()
#------------------------------------------------------读取数据集------------------------------------------------------#
# 假设你的数据集已经被加载到一个名为 data 的 pandas DataFrame 中
dataset=pd.read_excel('/scratch/uceezzz/Project/Baiproject/Dataset/GH_GOOC_GO_0726.xlsx')#这个是150条的
# dataset=dataset.loc[(dataset['G(H)']<24) & (dataset['G(H)']>-22)]
feature=dataset.iloc[:,1:-3] #总共是27个特征哈  注意，这个地方是取不到-2的，只能取到-3
label=dataset.iloc[:,-1].values

scaler = StandardScaler()
feature = scaler.fit_transform(feature)

#------------------------------------------------------计算特征组合------------------------------------------------------#
#计算一些排列组合，5个里面至少选三个
import itertools
features = list(range(20))
valid_combinations1 = []
valid_combinations2 = []

#从前20个特征中的每个范围内选择3-5个特征
for i in range(3, 5):  # 选择的特征数量，从3到5, 16种组合,只修改这个地方来改变特征数量！！！！！！！！！
    for indices in itertools.combinations(range(5), i):  # 在每个范围内选择特征的索引组合
        combination = []
        for start in range(0, 20, 5):  # 20：总的范围  5：每个原子的范围
            combination.extend([start + idx for idx in indices])  # 添加符合要求的特征索引到组合中
        valid_combinations1.append(combination)

#从后10个特征中的每个范围内选择3-7个特征
items = list(range(20,27))  # 为了减少计算难度，这里我们只使用了后7个特征
for i in range(4, 8):  # 选择的特征个数范围
    combination = list(itertools.combinations(items, i))
    valid_combinations2.extend(combination)
list_of_lists = []
for t in valid_combinations2:
    list_of_lists.append(list(t))
valid_combinations2 = list_of_lists
#把两种组合合并,总共是
all_combinations = []

for comb1 in valid_combinations1:
    for comb2 in valid_combinations2:
        all_combinations.append(comb1 + comb2)


#------------------------------------------------------参数寻优------------------------------------------------------#

# 指定你要使用的回归器及其参数网格
regressors = {
    "SVR": (SVR(), {
        "kernel": ["linear", "rbf"],
        "C": [0.1, 1, 10],
        "epsilon": [0.01, 0.1, 1],
    }),
    "MLPRegressor": (MLPRegressor(), {
        "hidden_layer_sizes": [(50,), (100,)],
        "activation": ["relu", "tanh"],
        "learning_rate": ["constant", "invscaling"],
        "max_iter": [200,300, 500],
    }),
}

all_feature_combinations = all_combinations

kf = KFold(n_splits=5, shuffle=True, random_state=0)

best_r2_score = -1e9
best_rmse = 1e9
best_regressor = None
best_params = None
best_feature_combination = None
total_combinations = len(all_feature_combinations)

with open('/scratch/uceezzz/Project/Baiproject/Code/07.26/Best_pre_GH.txt', 'w') as file:
    for idx, feature_combination in tqdm(enumerate(all_feature_combinations), total=len(all_feature_combinations)):
        selected_features = feature[:, feature_combination]
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"已消耗的时间: {elapsed_time:.2f} 秒")
        for name, (regressor, params) in regressors.items():

            rmse_scorer = make_scorer(lambda y_true, y_pred: sqrt(mean_squared_error(y_true, y_pred)), greater_is_better=False)

            grid_search = GridSearchCV(regressor, params, scoring={'r2': 'r2', 'rmse': rmse_scorer}, cv=kf, n_jobs=-1,refit='r2')
            
            grid_search.fit(selected_features, label)

            best_estimator = grid_search.best_estimator_
            best_params_combination = grid_search.best_params_
            # 获取cv_results
            cv_results = grid_search.cv_results_
            
            mean_r2 = cv_results['mean_test_r2'][grid_search.best_index_]
            mean_rmse = -cv_results['mean_test_rmse'][grid_search.best_index_]  # 记得取负值，因为我们之前设定为越小越好
            
            # 计算r2和rmse的标准差
            std_r2 = cv_results['std_test_r2'][grid_search.best_index_]
            std_rmse = cv_results['std_test_rmse'][grid_search.best_index_] 

            # 判断是否是最佳组合
            if mean_r2 > best_r2_score or (mean_r2 == best_r2_score):
                best_r2_score = mean_r2
                best_rmse = mean_rmse
                best_regressor = name
                best_params = best_params_combination
                best_feature_combination = feature_combination
                best_r2_var = std_r2
                best_rmse_var = std_rmse
                    
                file.seek(0)
                file.truncate()
                
                file.write("最佳R2分数: {}\n".format(best_r2_score))
                file.write("最佳RMSE: {}\n".format(best_rmse))
                file.write("最佳特征组合: {}\n".format(best_feature_combination))
                file.write("最佳模型: {}\n".format(best_regressor))
                file.write("最佳参数: {}\n".format(best_params))
                file.write("R^2方差: {}\n".format(best_r2_var))  
                file.write("RMSE方差: {}\n".format(best_rmse_var))
                file.flush()





