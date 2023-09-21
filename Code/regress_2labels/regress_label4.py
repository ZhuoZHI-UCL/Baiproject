import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import time
import warnings
from sklearn.exceptions import ConvergenceWarning
# 过滤掉 ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
start_time = time.time()
# 假设你的数据集已经被加载到一个名为 data 的 pandas DataFrame 中
dataset=pd.read_excel('../../Dataset/label_regression.xlsx')#这个是150条的
dataset=dataset.loc[(dataset['Ebin']<24) & (dataset['Ebin']>-22)]
feature=dataset.iloc[:,2:15]
label=dataset.iloc[:,-1].values

# 指定你要使用的回归器及其参数网格
regressors = {
    "XGBRegressor": (XGBRegressor(), {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.3],
        "max_depth": [3, 6, 9],
        "n_jobs": [10],
        
    }),
    "RandomForestRegressor": (RandomForestRegressor(), {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 6, 9],
    }),
    "AdaBoostRegressor": (AdaBoostRegressor(), {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.3],
    }),
    "BaggingRegressor": (BaggingRegressor(), {
        "n_estimators": [10, 50, 100],
        "max_samples": [0.5, 1.0],
    }),
    "GradientBoostingRegressor": (GradientBoostingRegressor(), {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.3],
        "max_depth": [3, 6, 9],
    }),
    "SVR": (SVR(), {
        "kernel": ["linear", "rbf"],
        "C": [0.1, 1, 10],
        "epsilon": [0.01, 0.1, 1],
    }),
    "MLPRegressor": (MLPRegressor(), {
        "hidden_layer_sizes": [(50,), (100,), (200,)],
        "activation": ["relu", "tanh"],
        "learning_rate": ["constant", "invscaling"],
        "max_iter": [200, 300, 500],
    }),
}


# 定义一个函数来计算特征组合
def feature_combinations(features):
    feature_names = features.columns
    all_combinations = []
    for i in range(1, len(feature_names) + 1):
        all_combinations.extend(itertools.combinations(feature_names, i))
    return all_combinations


all_feature_combinations = feature_combinations(feature)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

best_r2_score = -1e9
best_rmse = 1e9
best_regressor = None
best_params = None
best_feature_combination = None
total_combinations = len(all_feature_combinations)
for idx,feature_combination in enumerate(all_feature_combinations):
    selected_features = feature[list(feature_combination)]
    scaler = StandardScaler()
    selected_features = scaler.fit_transform(selected_features)
    print(f"进度: 特征组合 {idx + 1} / {total_combinations} ({round((idx + 1) / total_combinations * 100, 2)}%)")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"已消耗的时间: {elapsed_time:.2f} 秒")
    for name, (regressor, params) in regressors.items():
        print(f"正在尝试模型: {name}")
        grid_search = GridSearchCV(regressor, params, scoring="r2", cv=kf, n_jobs=-1)
        grid_search.fit(selected_features, label)

        best_estimator = grid_search.best_estimator_
        best_params_combination = grid_search.best_params_
        r2 = grid_search.best_score_

        # 计算 RMSE
        rmse = 0
        for train_index, test_index in kf.split(selected_features):
            X_train, X_test = selected_features[train_index], selected_features[test_index]
            y_train, y_test = label[train_index], label[test_index]
            best_estimator.fit(X_train, y_train)
            y_pred = best_estimator.predict(X_test)
            rmse += np.sqrt(mean_squared_error(y_test, y_pred))
        rmse /= kf.get_n_splits()

        # 判断是否是最佳组合
        if r2 > best_r2_score or (r2 == best_r2_score and rmse < best_rmse):
            best_r2_score = r2
            best_rmse = rmse
            best_regressor = name
            best_params = best_params_combination
            best_feature_combination = feature_combination

print("最佳R2分数: ", best_r2_score)
print("最佳RMSE: ", best_rmse)
print("最佳特征组合: ", best_feature_combination)
print("最佳模型: ", best_regressor)
print("最佳参数: ", best_params)

