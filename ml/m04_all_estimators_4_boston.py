#보스턴 하우징 집값

from sklearn.datasets import load_boston
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split        # train data, test data 분리
from sklearn.metrics import r2_score



# 1. 데이터 분석

datasets = load_boston()

x = datasets.data   
y = datasets.target


print(np.min(x), np.max(x))             # 0.0   711.0

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=66, test_size=0.8)


# print(x.shape)
# print(x_train.shape)
# print(x_test.shape)

from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer

scaler = PowerTransformer()

scaler.fit(x_train)                                
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)                    






# 3. 학습

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

allAlgorithms = all_estimators(type_filter='classifier')

for ( name , algorithm ) in allAlgorithms:
    try :
        model = algorithm()

        model.fit(x_train, y_train)

        y_predict = model.predict(x_test)
        acc = accuracy_score(y_test, y_predict)
        print(name, '의 정답률 : ', acc)
    except:
        print(name, '는 없음')
        continue