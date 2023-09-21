#这个是新增了150条数据,
#还是简单的回归出来标签值
#最好的结果是用MLP R2 0.85 RMSE 0.38
#其实只用了100条数据

import pandas as pd
from sklearn.preprocessing import scale,minmax_scale
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error #均方误差
from sklearn.metrics import mean_absolute_error #平方绝对误差
from sklearn.metrics import r2_score#R squar
import math
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn import datasets,decomposition,manifold
from sklearn.model_selection import KFold

def performance_evaluate_regression(x_test, y_test):
    '''
    calculate evaluation metrix and plot the predict result:
                       evaluation metrix:  1. RMSE
                                           2. MAE
                                           3. R2 score
                                    plot:  1. the image of true value and predict value (sorted by order)
                                           2. the histogram of true value and predict value (distribution)
    :param x_test: x_test
    :param y_test: y_test
    :return: none
    '''
    #get the predict result
    y_pred = model.predict(x_test)
    y_test=np.array(y_test)
    y_pred=np.array(y_pred)
    #calculate MSE, RMSE and R2 score as the regression evaluation metrix
    # MSE = mean_squared_error(y_test, y_pred)
    # RMSE = math.sqrt(MSE)
    # print('RMSE is %.2f' % RMSE)
    # MAE = mean_absolute_error(y_test, y_pred)
    # print('MAE is %.2f' % MAE)
    # r2score = r2_score(y_test, y_pred)
    # print('r2score is %.2f' % r2score)

    # sort the predict and true value by order
    order = np.argsort(y_test.flatten())
    y_pred, y_test = y_pred[order], y_test[order]

    # plot the sorted predict and true value
    plt.figure()

    plt.scatter(range(0, len(y_pred)), y_pred,
                label='Predicted value',
                # s = 20,
                color='royalblue',
                marker='.')
    plt.plot(y_test, label='True value', color='darkorange')
    # plt.ylim(0,17)

    # from matplotlib.pyplot import MultipleLocator
    # x = range(0, len(y_pred), 1)
    # x_major_locator = MultipleLocator(1)
    # ax = plt.gca()
    # ax.yaxis.set_major_locator(x_major_locator)  # x-axis is divided by 1
    # plt.grid()
    # plt.scatter(x, y_pred, label='Prediction')
    plt.legend()
    plt.savefig("../Save/True_pre.jpg")

    #plot the histogram of true value and predict value
    plt.figure()
    bins = range(5, 20, 1)
    plt.hist(y_pred, bins, alpha=0.8, label='y_pre')
    plt.hist(y_test, bins, alpha=0.8, label='y_true')
    plt.legend()
    plt.savefig("../Save/pre_hist.jpg")
print('All started')
#读取原始数据
dataset=pd.read_excel('../Dataset/150pieces.xlsx')#这个是150条的
#只保留标签(-20,9)之间的
dataset=dataset.loc[(dataset['G(H)']<9) & (dataset['G(H)']>-19)]
print(len(dataset))
dataset=dataset[0:100]

#分割特征与标签
feature=dataset.iloc[:,2:15]
label=dataset.iloc[:,-1]
columns_name=list(feature.columns)

#画一下标签的直方图
#画一下标签
# 使用sorted()函数对列表进行排序
sorted_numbers = sorted(label)
# 绘制直方图
plt.hist(sorted_numbers, bins=20, alpha=0.5, color='b')
plt.xlabel('Number')
plt.ylabel('Frequency')
plt.title('Sorted Numbers')
plt.show()



#看一下皮尔逊系数，筛选一下特征
# corr=dataset.corr(method='pearson', min_periods=1)
# print(corr[(corr>0.8)&(corr!=1)])

# First ionization energy-Inm  和  Pauling electronegativity-E ele_m  0.954
# 0.00497                                          0.11978
#
# Covalent radius(pm)-rnm 和 Atomic number Nnm 0.83
# -0.3965                                        -0.27

#特征中删去First ionization energy-Inm 和 Atomic number Nnm
# feature.drop(columns = ['First ionization energy-Inm','Atomic number_Nnm'],inplace = True)
#只挑选特征重要性排名高的
feature=feature[[

'Atomic number_Nnm',
'Covalent radius(pm)-rm',
'electron affinity(eV)-E_aff_nm',

'Covalent radius(pm)-rnm',
# 'Atomic number_Nm',
'Pauling electronegativity-E_ele_nm',
'd electrons-Nd',
# # 'Pauling electronegativity-E_ele_m',
# 'First ionization energy-Inm',
# 'N_same'
]]
#排序出来的


#特征归一化
feature=scale(feature)
label=scale(label)
# label=np.array(label)

# feature=minmax_scale(feature)
# label=minmax_scale(label)
#模型
#######################xgb
model = XGBRegressor(
    # objective=fair_obj, #use the customized loss function
    # tree_method='gpu_hist', #if use the gpu, 'gpu_hist' can accelerate the speed
    n_estimators=10000,
    max_depth=18,
    learning_rate=0.1,
    gamma=0.9,
    min_child_weight=5,
    max_delta_step=0.9,
    subsample=0.9,
    colsample_bytree=1,
    reg_alpha=0.5,
    reg_lambda=1,
    nthread=-1
    )

random_state=0
fold_num=5
kf = KFold(n_splits=fold_num, shuffle=True, random_state=random_state)
feature_5_folder=kf.split(feature)
sum1=sum2=0

#五折交叉验证
for train_index, test_index in kf.split(feature):
    x_train,x_test=feature[train_index],feature[test_index]
    y_train,y_test=label[train_index],label[test_index]
    x_validation=x_test
    y_validation=y_test
    # train the model
    eval_set = [(x_train, y_train), (x_validation, y_validation)]
    model.fit(x_train, y_train,
              early_stopping_rounds=5,
              eval_set=eval_set,
              eval_metric="rmse"
              )

    # evaluate the model
    y_pred = model.predict(x_test)
    y= np.array(y_test)
    y_pred = np.array(y_pred)
    r2score1 = r2_score(y, y_pred)
    sum1=sum1+r2score1


    y_pred = model.predict(x_train)
    y= np.array(y_train)
    y_pred = np.array(y_pred)
    r2score2 = r2_score(y, y_pred)
    sum2 = sum2 + r2score2
r2score_test=sum1/fold_num
r2score_train=sum2/fold_num
print('xgb r2score is %.2f and %.2f' % (r2score_test, r2score_train))
performance_evaluate_regression(x_test,y_test)

#特征重要性排行
# importances=model.feature_importances_
# indices = np.argsort(importances)[::-1]
# feat_labels = columns_name
# print("Feature ranking:")
# for f in range(x_train.shape[1]):
#     print("%d. feature no:%d feature name:%s (%f)" % (f + 1, indices[f], feat_labels[indices[f]], importances[indices[f]]))
# print (">>>>>", importances)


#测试一下其他模型的性能
models=[
            RandomForestRegressor(),
            AdaBoostRegressor(),
            GradientBoostingRegressor(),
            # GradientBoostingRegressor(subsample=0.9,learning_rate=0.1,n_estimators=130,max_depth=2),
            BaggingRegressor(),
            LinearRegression( normalize=True),
            SVR(C=3,gamma=0.07),#设置gamma忽略可以得到r2=0.81
            MLPRegressor(hidden_layer_sizes=56,max_iter=380,
                         random_state=3,
                         learning_rate_init= 0.0011,
                         alpha=0.1,momentum=0.85)
        ]
models_str=[
'RandomForestRegressor','AdaBoostRegressor','GradientBoostingRegressor','BaggingRegressor','LinearRegression',
'SVR','MLP'
# 'MLPRegressor'
]
for name, model in zip(models_str, models):
    sum3=sum4=sum5=0
    for train_index, test_index in kf.split(feature):
        x_train, x_test = feature[train_index], feature[test_index]
        y_train, y_test = label[train_index], label[test_index]

        model = model  # 建立模型
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)
        y = np.array(y_test)
        y_pred = np.array(y_pred)
        r2score3 = r2_score(y, y_pred)
        sum3 = sum3 + r2score3

        MSE = mean_squared_error(y, y_pred)
        RMSE = math.sqrt(MSE)
        sum5 = sum5 + MSE

        y_pred = model.predict(x_train)
        y = np.array(y_train)
        y_pred = np.array(y_pred)
        r2score4 = r2_score(y, y_pred)
        sum4 = sum4 + r2score4


    RMSE=sum5/fold_num
    r2score_test=sum3/fold_num
    r2score_train = sum4 / fold_num
    print(name +'r2score is %.2f and %.2f MSE is %.2f' % (r2score_test,r2score_train,RMSE))



print('All finished')