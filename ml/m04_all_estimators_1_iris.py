import numpy as np
from sklearn.datasets import load_iris
from sklearn.datasets import load_boston
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split        # train data, test data 분리
from sklearn.metrics import r2_score

from sklearn.utils import all_estimators


# 1. 데이터 분석

datasets = load_iris()

x = datasets.data
y = datasets.target

# print(x.shape, y.shape) # (150, 4) (150,)
# print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=66, \
    test_size=0.7)


from sklearn.preprocessing import MinMaxScaler, StandardScaler                        
scaler = StandardScaler()
scaler.fit(x_train)                              
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)                   




# 2. 모델 구성

# allAlgorithms = all_estimators(type_filter='classifier')
allAlgorithms = all_estimators(type_filter='classifier')

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


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







'''
# 3. 컴파일, 훈련

model.fit(x_train, y_train)



print("============== 평가, 예측 ===============")
# 4. 예측 및 평가
results = model.score(x_test, y_test)
print(results)


from sklearn.metrics import r2_score, accuracy_score

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print("accuracy_score : ", acc)




print("============== 예측 ===============")

print(y_test[:5])
y_predict2 = model.predict(x_test[:5])
print(y_predict2)
'''
