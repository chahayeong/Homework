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


from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV

# n_splits : 데이터 셋 전체 평가및 훈련 / test와 train 나누는 비율
n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)       




parameters = [
    {"C":[1, 10, 100, 1000], "kernel":["linear"]},
    {"C":[1, 10, 100], "kernel":["rbf"], "gamma":[0.001, 0.0001]},
    {"C":[1, 10, 100, 1000], "kernel":["sigmoid"], "gamma":[0.001, 0.0001]}
]


x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=66, \
    test_size=0.7)



# 2. 모델 구성

from sklearn.svm import SVC
# model = SVC()

model = GridSearchCV(SVC(), parameters, cv=kfold)       # fit 지원


# 3. 훈련

model.fit(x, y)


# 4. 예측 및 평가

print("최적의 매개변수 : ", model.best_estimator_)
print("best_score_ : ", model.best_score_)

print("model.score : ", model.score(x_test, y_test))

y_predict = model.predict(x_test)
print("정답률 : ", accuracy_score(y_test, y_predict))


'''

'''
