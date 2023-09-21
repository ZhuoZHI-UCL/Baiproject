#这个特征是目前最好的哈
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.preprocessing import scale,minmax_scale
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error #均方误差
from sklearn.metrics import mean_absolute_error #平方绝对误差
from sklearn.metrics import r2_score#R squar
import math
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
    # order = np.argsort(y_test.flatten())
    # y_pred, y_test = y_pred[order], y_test[order]
    #
    # # plot the sorted predict and true value
    # plt.figure()
    #
    # plt.scatter(range(0, len(y_pred)), y_pred,
    #             label='Predicted value',
    #             # s = 20,
    #             color='royalblue',
    #             marker='.')
    # plt.plot(y_test, label='True value', color='darkorange')
    # plt.ylim(0,17)

    # from matplotlib.pyplot import MultipleLocator
    # x = range(0, len(y_pred), 1)
    # x_major_locator = MultipleLocator(1)
    # ax = plt.gca()
    # ax.yaxis.set_major_locator(x_major_locator)  # x-axis is divided by 1
    # plt.grid()
    # plt.scatter(x, y_pred, label='Prediction')
    # plt.legend()
    # plt.savefig("../Save/True_pre.jpg")

    #plot the histogram of true value and predict value
    # plt.figure()
    # bins = range(5, 20, 1)
    # plt.hist(y_pred, bins, alpha=0.8, label='y_pre')
    # plt.hist(y_test, bins, alpha=0.8, label='y_true')
    # plt.legend()
    # plt.savefig("../Save/pre_hist.jpg")
print('All started')
#读取原始数据
dataset=pd.read_excel('/scratch/uceezzz/Project/Baiproject/Dataset/G(H).xlsx')
dataset_test = pd.read_excel('/scratch/uceezzz/Project/Baiproject/Dataset/0707.xlsx')
#只保留标签(-20,9)之间的
# dataset=dataset.loc[(dataset['G(H)']<5) & (dataset['G(H)']>-5)]#100个样本的
# dataset=dataset.loc[(dataset['G(H)']<3.3) & (dataset['G(H)']>-3.3)]#60个样本

print(len(dataset))
# dataset=dataset[0:120]

#分割特征与标签
feature=dataset.iloc[:,1:-2]
feature_test=dataset_test.iloc[:,1:]
label=dataset.iloc[:,-1].values
columns_name=list(feature.columns)

#前四个范围内选择的相对序号m
#列的范围是 1-5 6-10 11-15 16-20 21-30
#26, 29 28

#调好的参数 到0.8,150个样本的
# f= [0, 1, 3,4,
#      5, 6, 8,9,
#      10, 11, 13,14,
#      15, 16, 18,19
#     ,23, 25,29,26,28,27,21,
#      ]

# f = [25, 13, 23, 18, 15, 29, 5, 10, 6, 20,
#      26, 7,
#      11, 12,
#      28, 27,
#      17, 8,
#      16, 9, 21,
#      3, 0,
#      19,
#      # 4,
#      # 22, 14, 24, 1, 2
#      ]
#
# f= [0, 1, 2,
#      5, 6, 7,
#      10, 11, 14,
#      15, 16, 17,
#      21, 23, 24, 25
#      ]

f = [1, 2, 4, 5, 6, 7, 10, 11, 14, 15, 16, 17, 23, 24, 25, 26]
f_test = [1, 2, 4, 5, 6, 7, 10, 11, 14, 15, 16, 17, 23, 24, 25,26]

feature = feature.iloc[:,f]
feature_test = feature_test.iloc[:,f_test]


#特征归一化
scaler = StandardScaler()
feature = scaler.fit_transform(feature)
feature_test = scaler.fit_transform(feature_test)
# label=scale(label)

random_state=0
fold_num=5
kf = KFold(n_splits=fold_num, shuffle=True, random_state=random_state)
feature_5_folder=kf.split(feature)
sum1=sum2=0


#测试一下其他模型的性能
models=[
            RandomForestRegressor(),
            AdaBoostRegressor(),
            # GradientBoostingRegressor(),
            GradientBoostingRegressor(subsample=0.9,learning_rate=0.1,n_estimators=130,max_depth=2),
            BaggingRegressor(),
            LinearRegression(),
            # SVR(C=6,kernel='linear'),#设置gamma忽略可以得到r2=0.81 #150个样本的 r20.79
            SVR(C=6,kernel='linear'),
            MLPRegressor(activation='relu', max_iter=500,hidden_layer_sizes=50,learning_rate='invscaling')
        ]
models_str=[
'RandomForestRegressor',
'AdaBoostRegressor','GradientBoostingRegressor','BaggingRegressor','LinearRegression',
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
        diff = abs(y - y_pred)
        # num=6#100个点
        num = 0
        max_index = np.argpartition(-diff, num)[:num]
        y_pred = np.delete(y_pred, max_index)
        y = np.delete(y, max_index)


        r2score3 = r2_score(y, y_pred)
        sum3 = sum3 + r2score3

        RMSE = math.sqrt(mean_squared_error(y, y_pred))
        sum5 = sum5 + RMSE

        y_pred = model.predict(x_train)
        y = np.array(y_train)
        y_pred = np.array(y_pred)
        diff = abs(y - y_pred)
        max_index = np.argpartition(-diff, num)[:num]
        y_pred = np.delete(y_pred, max_index)
        y = np.delete(y, max_index)
        r2score4 = r2_score(y, y_pred)
        sum4 = sum4 + r2score4

        y_pred = model.predict(feature_test)
        print(y_pred)


    RMSE=sum5/fold_num
    r2score_test=sum3/fold_num
    r2score_train = sum4 / fold_num
    print(name +'r2score is %.2f and %.2f RMSE is %.2f' % (r2score_test,r2score_train,RMSE))

#进行测试


print('All finished')