import numpy as np
import warnings 
warnings.filterwarnings(action='ignore')

# 1. data
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier

datasets = load_diabetes()
x = datasets.data
y = datasets.target



n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, 
                random_state=66)

# 2. model 구성

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

parameters = [{
    "n_estimators": [100, 200], # =. epochs, default = 100
    "max_depth": [6, 8, 10, 12],
    "min_samples_leaf": [3, 5, 7, 10],
    "min_samples_split": [2, 3, 5, 10],
    "n_jobs": [-1] # =. qauntity of cpu; -1 = all
}]

model = RandomizedSearchCV(RandomForestRegressor(), parameters, cv=kfold, verbose=1)

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
totla time :  9.462167978286743
Best estimator :  RandomForestRegressor(max_depth=10, min_samples_leaf=10, n_estimators=200,
                      n_jobs=-1)
Best score  : 0.45722786323571435
'''