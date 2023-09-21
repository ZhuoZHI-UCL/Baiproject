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
from joblib import load
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
all_combinations = load('file_6.joblib')

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

with open('/scratch/uceezzz/Project/Baiproject/Code/06.18/GH/record/6.txt', 'w') as file:
    for idx, feature_combination in tqdm(enumerate(all_feature_combinations), total=len(all_feature_combinations)):
        selected_features = feature[:, feature_combination]
        # print(f"进度: 特征组合 {idx + 1} / {total_combinations} ({round((idx + 1) / total_combinations * 100, 2)}%)")
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"已消耗的时间: {elapsed_time:.2f} 秒")
        for name, (regressor, params) in regressors.items():
            # print(f"正在尝试模型: {name}")
            grid_search = GridSearchCV(regressor, params, scoring="r2", cv=kf, n_jobs=-1)
            grid_search.fit(selected_features, label)

            best_estimator = grid_search.best_estimator_
            best_params_combination = grid_search.best_params_
            r2 = grid_search.best_score_

            # 计算 RMSE,太耗时间，给去掉
            # rmse = 0
            # for train_index, test_index in kf.split(selected_features):
            #     X_train, X_test = selected_features[train_index], selected_features[test_index]
            #     y_train, y_test = label[train_index], label[test_index]
            #     best_estimator.fit(X_train, y_train)
            #     y_pred = best_estimator.predict(X_test)
            #     rmse += np.sqrt(mean_squared_error(y_test, y_pred))
            # rmse /= kf.get_n_splits()

            # 判断是否是最佳组合
            if r2 > best_r2_score or (r2 == best_r2_score):
                best_r2_score = r2
                # best_rmse = rmse
                best_regressor = name
                best_params = best_params_combination
                best_feature_combination = feature_combination

                    
                file.seek(0)
                file.truncate()
                
                file.write("最佳R2分数: {}\n".format(best_r2_score))
                # file.write("最佳RMSE: {}\n".format(best_rmse))
                file.write("最佳特征组合: {}\n".format(best_feature_combination))
                file.write("最佳模型: {}\n".format(best_regressor))
                file.write("最佳参数: {}\n".format(best_params))
                file.flush()


print("最佳R2分数: ", best_r2_score)
# print("最佳RMSE: ", best_rmse)
print("最佳特征组合: ", best_feature_combination)
print("最佳模型: ", best_regressor)
print("最佳参数: ", best_params)




