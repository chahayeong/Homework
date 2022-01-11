from sklearn.datasets import load_boston
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split        # train data, test data 분리
from sklearn.metrics import r2_score

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

# 1. 데이터 분석

datasets = load_boston()

x = datasets.data   
y = datasets.target

# 2. model 구성

from sklearn.svm import LinearSVC, SVC, SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
from sklearn.pipeline import Pipeline, make_pipeline

parameters = [{
    "randomforestregressor__max_depth": [6, 8, 10, 12],
    "randomforestregressor__min_samples_leaf": [3, 5, 7],
    "randomforestregressor__min_samples_split": [2, 3, 5, 10],
}] # model and param connection : model__param

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, 
                random_state=66)

pipe = make_pipeline(MinMaxScaler(), RandomForestRegressor())

# model = RandomizedSearchCV(pipe, parameters, cv=kfold, verbose=1)
model = GridSearchCV(pipe, parameters, cv=kfold, verbose=1)

# 3. 컴파일 훈련
import time
st = time.time()
model.fit(x, y)
et = time.time() - st

# 4. 평가 예측

print('totla time : ', et)
print('Best estimator : ', model.best_estimator_)
print('Best score  :', model.best_score_)


'''
totla time :  32.55052399635315
Best estimator :  Pipeline(steps=[('minmaxscaler', MinMaxScaler()),
                ('randomforestregressor',
                 RandomForestRegressor(max_depth=12, min_samples_leaf=3))])
Best score  : 0.8596805009289957
'''