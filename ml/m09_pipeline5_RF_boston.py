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


# 1. 데이터 분석

datasets = load_boston()

x = datasets.data   
y = datasets.target




x_train, x_test, y_train, y_test = train_test_split(x, y,
      test_size=0.1, shuffle=True, random_state=66)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer

# 2. model 구성
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import Pipeline, make_pipeline

model = make_pipeline(MinMaxScaler(), RandomForestRegressor(), )
# model = LogisticRegression()

# 3. 컴파일 훈련
import time
st = time.time()
model.fit(x_train, y_train)
et = time.time() - st

# 4. 평가 예측
from sklearn.metrics import r2_score, accuracy_score

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

print('r2_score :', r2)
print('model_score :', model.score(x_test, y_test))


'''
r2_score : 0.923014102992468
model_score : 0.923014102992468
'''