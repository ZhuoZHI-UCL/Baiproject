#这个文件是用150条然后进行回归，回归的标签有两个,这个文件是针对第一个标签的

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
#读取训练原始数据
dataset=pd.read_excel('../../Dataset/two_labels_regression.xlsx')#这个是150条的
dataset=dataset.loc[(dataset['Ef']<35) & (dataset['Ef']>-11)]
#读取最终要预测的所有值
dataset_pre=pd.read_excel('../../Dataset/data_20000.xlsx')
#要写入的新的excel
dataset_new=pd.read_excel('../../Save/regress_2labels.xlsx')

feature_pre=dataset_pre[[

'Atomic number_Nnm',
'Covalent radius(pm)-rm',
'electron affinity(eV)-E_aff_nm',

'Covalent radius(pm)-rnm',
# 'Atomic number_Nm',
'Pauling electronegativity-E_ele_nm',
'd electrons-Nd',
# 'Pauling electronegativity-E_ele_m',
'First ionization energy-Inm',
# 'N_same'
]]
feature_pre=scale(feature_pre)

#分割特征与标签
feature=dataset.iloc[:,2:15]
label=dataset.iloc[:,-2].values
# label_2=dataset.iloc[:,-1].values
columns_name=list(feature.columns)

feature=feature[[

'Atomic number_Nnm',
'Covalent radius(pm)-rm',
'electron affinity(eV)-E_aff_nm',

'Covalent radius(pm)-rnm',
# 'Atomic number_Nm',
'Pauling electronegativity-E_ele_nm',
'd electrons-Nd',
# 'Pauling electronegativity-E_ele_m',
'First ionization energy-Inm',
# 'N_same'
]]

#特征归一化
feature=scale(feature)
label=scale(label)
random_state=0
fold_num=5
kf = KFold(n_splits=fold_num, shuffle=True, random_state=random_state)
feature_5_folder=kf.split(feature)

#所有的模型
models=[    XGBRegressor(),
            RandomForestRegressor(),
            AdaBoostRegressor(),
            BaggingRegressor(),
            GradientBoostingRegressor(),
            SVR(max_iter=400),
            MLPRegressor(hidden_layer_sizes=100,max_iter=240)
        ]
models_str=[
'XGB','RandomForest','AdaBoost','GradientBoosting','Bagging',
'SVM','MLP'
]
label1_sum=np.zeros(len(feature_pre))#最终把这个取平均
for name, model in zip(models_str, models):
    sum1=sum2=sum3=sum4=0
    # print(model.__class__.__name__)
    for train_index, test_index in kf.split(feature):
        x_train, x_test = feature[train_index], feature[test_index]
        y_train, y_test = label[train_index], label[test_index]

        model = model  # 建立模型
        model.fit(x_train, y_train)

        if model.__class__.__name__=='MLPRegressor':
            print('MLP model is the best and the label1 of 20000 data is pred')
            label1_20000=model.predict(feature_pre)
            label1_sum=label1_sum+label1_20000

        #测试集
        y_pred = model.predict(x_test)
        y = np.array(y_test)
        y_pred = np.array(y_pred)
        rmse = math.sqrt(mean_squared_error(y_test, y_pred))
        R2 = r2_score(y_test, y_pred)
        sum1 = sum1 + rmse
        sum2=sum2+R2

        #训练集
        y_pred = model.predict(x_train)
        y = np.array(y_train)
        y_pred = np.array(y_pred)
        rmse = math.sqrt(mean_squared_error(y_train, y_pred))
        R2 = r2_score(y_train, y_pred)
        sum3=sum3+rmse
        sum4=sum4+R2

    rmse_test=sum1/fold_num
    rmse_train=sum3/fold_num
    R2_test=sum2/fold_num
    R2_train=sum4/fold_num

    # print(name +'acc_test is %.2f, f1_test is %.2f **** acc_train is %.2f, f1_train is %.2f' % (acc_test,f1_test,acc_train,f1_train))
    print(name +' R2_test is %.2f ****  R2_train is %.2f' % (R2_test,R2_train))
    print(name + ' rmse_test is %.2f ****  rmse_train is %.2f' % (rmse_test, rmse_train))
label1_sum=label1_sum/5


#把label1_sum写入excel并命名为Ef
dataset_new['EF']=label1_sum
dataset_new.to_excel('../../Save/regress_2labels.xlsx', index=False)
print('All finished')

