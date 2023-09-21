#这个文件是用来网格搜索GH_150最后一个变量的回归器的最优参数 05.22
import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import r2_score, mean_squared_error
#from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
#from xgboost import XGBRegressor
#import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import time
import warnings
from sklearn.exceptions import ConvergenceWarning
# 过滤掉 ConvergenceWarning

from sklearn.exceptions import ConvergenceWarning
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    start_time = time.time()
    # 假设你的数据集已经被加载到一个名为 data 的 pandas DataFrame 中
    dataset=pd.read_excel('/scratch/uceezzz/Project/Baiproject/Dataset/0522.xlsx')#这个是150条的
    # dataset=dataset.loc[(dataset['G(H)']<24) & (dataset['G(H)']>-22)]
    feature=dataset.iloc[:,1:-2]
    label=dataset.iloc[:,-1].values

    # 指定你要使用的回归器及其参数网格
    regressors = {
        # "XGBRegressor": (XGBRegressor(), {
        #     "n_estimators": [50, 100, 200],
        #     "learning_rate": [0.01, 0.1, 0.3],
        #     "max_depth": [3, 6, 9],
        # }),
        # "RandomForestRegressor": (RandomForestRegressor(), {
        #     "n_estimators": [50, 100, 200],
        #     "max_depth": [3, 6, 9],
        # }),
        # "AdaBoostRegressor": (AdaBoostRegressor(), {
        #     "n_estimators": [50, 100, 200],
        #     "learning_rate": [0.01, 0.1, 0.3],
        # }),
        # "BaggingRegressor": (BaggingRegressor(), {
        #     "n_estimators": [10, 50, 100],
        #     "max_samples": [0.5, 1.0],
        # }),
        # "GradientBoostingRegressor": (GradientBoostingRegressor(), {
        #     "n_estimators": [50, 100, 200],
        #     "learning_rate": [0.01, 0.1, 0.3],
        #     "max_depth": [3, 6, 9],
        # }),
        "SVR": (SVR(), {
            "kernel": ["linear", "rbf"],
            "C": [0.1, 1, 10],
            "epsilon": [0.01, 0.1, 1],
     
        }),
        "MLPRegressor": (MLPRegressor(), {
            "hidden_layer_sizes": [(50,), (100,), (200,)],
            "activation": ["relu", "tanh"],
            "learning_rate": ["constant", "invscaling"],
            "max_iter": [300, 400, 500], 
        }),
    }


    # 定义一个函数来计算特征组合
    def feature_combinations(features):
        feature_names = features.columns

        # 定义五个特征范围
        ranges = [(0, 5), (5, 10), (10, 15), (15, 20), (20, len(feature_names)+1)]

        # 在每个范围内至少选择两列
        range_combinations = []
        for idx, r in enumerate(ranges):
            range_comb = []
            for i in range(2, min(r[1]-r[0]+1, len(feature_names)-r[0]+1)):
                range_comb.extend(itertools.combinations(range(r[0], r[1]), i))
            range_combinations.append((idx+1, range_comb))

        # 取前四个范围的笛卡尔积
        first_four_combinations = list(itertools.product(*[comb[1] for comb in range_combinations[:4]]))

        # 合并每个组合的序号和特征组合
        all_combinations = []
        for comb in first_four_combinations:
            combined_comb = tuple(itertools.chain(*comb))
            all_combinations.append(combined_comb)

        return all_combinations

    all_feature_combinations = feature_combinations(feature)

    kf = KFold(n_splits=5, shuffle=True, random_state=0)

    best_r2_score = -1e9
    best_rmse = 1e9
    best_regressor = None
    best_params = None
    best_feature_combination = None
    total_combinations = len(all_feature_combinations)
    def process_feature_combination(feature_combination):
        selected_features = feature.iloc[:, list(feature_combination)]
        if selected_features.shape[1] < 6:
            return None

        scaler = StandardScaler()
        selected_features = scaler.fit_transform(selected_features)

        best_r2_score = -1e9
        best_rmse = 1e9
        best_regressor = None
        best_params = None
        best_feature_combination = None

        kf = KFold(n_splits=5, shuffle=True, random_state=0)

        for name, (regressor, params) in regressors.items():
            grid_search = GridSearchCV(regressor, params, scoring="r2", cv=kf, n_jobs=-1)
            grid_search.fit(selected_features, label)

            best_estimator = grid_search.best_estimator_
            best_params_combination = grid_search.best_params_
            r2 = grid_search.best_score_

            rmse = np.mean([
                np.sqrt(mean_squared_error(label[test_index], best_estimator.predict(selected_features[test_index])))
                for train_index, test_index in kf.split(selected_features)
            ])

            if r2 > best_r2_score or (r2 == best_r2_score and rmse < best_rmse):
                best_r2_score = r2
                best_rmse = rmse
                best_regressor = name
                best_params = best_params_combination
                best_feature_combination = feature_combination

        return (best_r2_score, best_rmse, best_regressor, best_params, best_feature_combination)


    all_feature_combinations = feature_combinations(feature)
    total_combinations = len(all_feature_combinations)

    best_results = None
    print('cao n m!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    with ThreadPoolExecutor() as executor:
        results = executor.map(process_feature_combination, all_feature_combinations)
        for result in tqdm(executor.map(process_feature_combination, all_feature_combinations), total=total_combinations, desc="Processing"):
            print('c n m!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            if result is not None:
                if best_results is None or result[0] > best_results[0]:
                    best_results = result

    best_r2_score, best_rmse, best_regressor, best_params, best_feature_combination = best_results
    print("最佳R2分数: ", best_r2_score)
    print("最佳RMSE: ", best_rmse)
    print("最佳特征组合: ", best_feature_combination)
    print("最佳模型: ", best_regressor)
    print("最佳参数: ", best_params)
