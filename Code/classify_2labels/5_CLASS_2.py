#这个文件是用150条然后进行分类，分类的标签有两个,这个文件是针对第一个标签的
#评价指标我们就直接用sklearn 库的吧

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn import  svm
from sklearn import datasets,decomposition,manifold
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,f1_score
print('All started')
#读取训练原始数据
dataset=pd.read_excel('../Dataset/two_labels.xlsx')#这个是150条的
#读取最终要预测的所有值
dataset_pre=pd.read_excel('../Dataset/data_20000.xlsx')#这个是20000条的
#要写入的新的excel
dataset_new=pd.read_excel('../Save/pre_2labels.xlsx')

feature_pre=dataset_pre.iloc[:,2:15]
feature_pre=feature_pre[[

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
label=dataset.iloc[:,-1].values
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
label=np.array(label)
random_state=0
fold_num=5
kf = KFold(n_splits=fold_num, shuffle=True, random_state=random_state)
feature_5_folder=kf.split(feature)
#所有的模型
models=[    XGBClassifier(),
            RandomForestClassifier(),
            AdaBoostClassifier(),
            BaggingClassifier(),
            GradientBoostingClassifier(),
            svm.SVC(max_iter=200),
            MLPClassifier(hidden_layer_sizes=60)
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

        if model.__class__.__name__=='MLPClassifier':
            print('svm.SVC model is the best and the label1 of 20000 data is pred')
            label1_20000=model.predict(feature_pre)
            label1_sum=label1_sum+label1_20000

        #测试集
        y_pred = model.predict(x_test)
        y = np.array(y_test)
        y_pred = np.array(y_pred)
        # acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        # sum1=sum1+acc
        sum2=sum2+f1

        #训练集
        y_pred = model.predict(x_train)
        y = np.array(y_train)
        y_pred = np.array(y_pred)
        # acc = accuracy_score(y_train, y_pred)
        f1 = f1_score(y_train, y_pred)
        # sum3=sum3+acc
        sum4=sum4+f1




    # acc_test=sum1/fold_num
    f1_test=sum2/fold_num
    # acc_train=sum3/fold_num
    f1_train=sum4/fold_num

    # print(name +'acc_test is %.2f, f1_test is %.2f **** acc_train is %.2f, f1_train is %.2f' % (acc_test,f1_test,acc_train,f1_train))
    print(name +' f1_test is %.2f ****  f1_train is %.2f' % (f1_test,f1_train))

xxx=label1_sum
label1_sum=label1_sum/5

for i in range(len(label1_sum)):
    if label1_sum[i]<0.5:
        label1_sum[i]=0
    else:
        label1_sum[i] = 1

#把label1_sum写入excel并命名为Ef
dataset_new['Ebin-Ecoh']=label1_sum
dataset_new.to_excel('../Save/pre_2labels.xlsx', index=False)
print('All finished')
