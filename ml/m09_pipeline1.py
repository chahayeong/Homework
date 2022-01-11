import numpy as np
from sklearn.datasets import load_iris
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split        # train data, test data 분리
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score


# 1. 데이터 분석

datasets = load_iris()

x = datasets.data
y = datasets.target

print(x.shape, y.shape) # (150, 4) (150,)
print(y)



# train_test_split : train data와 test data 자동으로 분리 해주는 함수
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=66, \
    test_size=0.7)


from sklearn.preprocessing import MinMaxScaler, StandardScaler                        


# 2. 모델 구성


from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.pipeline import make_pipeline, Pipeline

model = make_pipeline(MinMaxScaler(), SVC())        # model + scaling



# 3. 컴파일, 훈련


model.fit(x_train, y_train)


#

# 4. 예측 및 평가


print("model.score : ", model.score(x_test, y_test))

y_predict = model.predict(x_test)
print("accuracy_score : ", accuracy_score(y_test, y_predict))



'''
model.score :  0.9619047619047619
accuracy_score :  0.9619047619047619
'''




