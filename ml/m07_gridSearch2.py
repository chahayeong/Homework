# m07_1 최적의 파라미터값을 가지고 
# 모델을 svc안에 c=1 , 커널 linear 파라미터 값


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
import warnings                         # warnings 무시
warnings.filterwarnings('ignore')       # warnings 무시
from sklearn.metrics import accuracy_score


# 1. 데이터 분석

datasets = load_iris()
x = datasets.data
y = datasets.target




# 1. data
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split


x_train, x_test, y_train, y_test = train_test_split(x, y,
      test_size=0.25, shuffle=True, random_state=1)

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, 
                random_state=66)

# 2. model 구성

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier



model = SVC(C=10, kernel='linear')

# 3. 컴파일 훈련
model.fit(x_train, y_train)

# 4. 평가 예측
from sklearn.metrics import accuracy_score

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('acc_score   :', acc)
print('model score :', model.score(x_test, y_test))


'''
acc_score   : 0.9736842105263158
model score : 0.9736842105263158
'''