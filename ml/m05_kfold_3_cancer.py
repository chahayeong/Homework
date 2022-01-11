import numpy as np
from sklearn.model_selection import train_test_split 
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.metrics import r2_score
from sklearn.datasets import load_breast_cancer


# 이진분류 모델

datasets = load_breast_cancer()

# 1. 데이터 분석
x = datasets.data
y = datasets.target


from sklearn.model_selection import train_test_split, KFold, cross_val_score

# n_splits : 데이터 셋 전체 평가및 훈련 / test와 train 나누는 비율
n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)    



# 2. 모델 구성

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier()
# model = LogisticRegression()
# model = DecisionTreeClassifier()
model = RandomForestClassifier()



# 4. 예측 및 평가

scores = cross_val_score(model, x, y, cv=kfold)     # fit ~ score (훈련~평가)
print("Acc : ", scores, "평균 Acc : ", round(np.mean(scores),4))



'''
LinearSVC : 평균 Acc :  0.8963
SVC : 평균 Acc :  0.921
KNeighborsClassifier : 평균 Acc :  0.928
LogisticRegression : 평균 Acc :  0.9403
DecisionTreeClassifier : 평균 Acc :  0.9244
RandomForestClassifier : 평균 Acc :  0.9684
'''