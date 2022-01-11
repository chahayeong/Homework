import numpy as np
from sklearn.datasets import load_wine

from sklearn.model_selection import train_test_split 
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier

# 1. 데이터 분석

datasets = load_wine()

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
totla time :  21.094472646713257
Best estimator :  Pipeline(steps=[('minmaxscaler', MinMaxScaler()),
                ('randomforestclassifier',
                 RandomForestClassifier(max_depth=10, min_samples_leaf=3,
                                        min_samples_split=5))])
Best_params_ : {'randomforestclassifier__max_depth': 10, 'randomforestclassifier__min_samples_leaf': 3, 'randomforestclassifier__min_samples_split': 5}
Best score  : 0.9887301587301588
'''
