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


# 1. 데이터 분석

datasets = load_iris()
x = datasets.data
y = datasets.target


from sklearn.model_selection import train_test_split, KFold, cross_val_score

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
        shuffle=True, random_state=66, train_size=0.8
)



# n_splits : 데이터 셋 전체 평가및 훈련 / test와 train 나누는 비율
n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)       



# 2. 모델 구성

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# model = LinearSVC()             
# model = KNeighborsClassifier() 
# model = LogisticRegression()     
# model = DecisionTreeClassifier() 
# model = RandomForestClassifier()    
model = SVC()



# 3. 컴파일, 훈련

# model.fit(x_train, y_train)




# 4. 예측 및 평가

scores = cross_val_score(model, x_train, y_train, cv=kfold)     # fit ~ score (훈련~평가)
print("Acc : ", scores, "평균 Acc : ", round(np.mean(scores),4))



'''
LinearSVC : 평균 Acc :  0.9667
KNeighborsClassifier : 평균 Acc :  0.96
LogisticRegression : 평균 Acc :  0.9667
DecisionTreeClassifier : 평균 Acc :  0.9467
RandomForestClassifier : 평균 Acc :  0.9467
SVC : 0.9667
'''

# x_train y_train 사용시 SVC  평균 Acc :  0.9583