import numpy as np
from sklearn.datasets import load_wine


from sklearn.model_selection import GridSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier

# 1. 데이터 분석

datasets = load_wine()

x = datasets.data
y = datasets.target


from sklearn.model_selection import GridSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, KFold, cross_val_score, train_test_split

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, 
                random_state=66)

# 2. model 구성

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

parameters = [{
    "n_estimators": [100, 200],
    "max_depth": [6, 8, 10, 12],
    "min_samples_leaf": [3, 5, 7, 10],
    "min_samples_split": [2, 3, 5, 10],
    "n_jobs": [-1] 
}]

model = RandomizedSearchCV(RandomForestClassifier(), parameters, cv=kfold, verbose=1)


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
totla time :  7.342184782028198
Best estimator :  RandomForestClassifier(max_depth=8, min_samples_leaf=3, n_estimators=200,
                       n_jobs=-1)
Best score  : 0.9831746031746033
'''