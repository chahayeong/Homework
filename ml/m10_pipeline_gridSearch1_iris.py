# 08.7

import numpy as np
import warnings 
warnings.filterwarnings(action='ignore')

# 1. data
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


datasets = load_iris()
x = datasets.data
y = datasets.target


# 2. model 구성

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
from sklearn.pipeline import Pipeline, make_pipeline

parameters = [{
    "randomforestclassifier__max_depth": [6, 8, 10, 12],
    "randomforestclassifier__min_samples_leaf": [3, 5, 7],
    "randomforestclassifier__min_samples_split": [2, 3, 5, 10],
}] # model and param connection : model__param

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, 
                random_state=66)

pipe = make_pipeline(MinMaxScaler(), RandomForestClassifier())

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
print('Best_params_ :', model.best_params_)
print('Best score  :', model.best_score_)


'''
totla time :  20.380706548690796
Best estimator :  Pipeline(steps=[('minmaxscaler', MinMaxScaler()),
                ('randomforestclassifier',
                 RandomForestClassifier(max_depth=12, min_samples_leaf=3,
                                        min_samples_split=10))])
Best_params_ : {'randomforestclassifier__max_depth': 12, 'randomforestclassifier__min_samples_leaf': 
3, 'randomforestclassifier__min_samples_split': 10}
Best score  : 0.9666666666666666
'''