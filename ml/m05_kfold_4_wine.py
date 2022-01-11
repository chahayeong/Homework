import numpy as np
from sklearn.datasets import load_wine


# 1. 데이터 분석

datasets = load_wine()

x = datasets.data
y = datasets.target



from sklearn.model_selection import train_test_split, KFold, cross_val_score

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)    



# 2. 모델 구성

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier()
# model = LogisticRegression()
# model = DecisionTreeClassifier()
# model = RandomForestClassifier()



# 4. 예측 및 평가

scores = cross_val_score(model, x, y, cv=kfold)  
print("Acc : ", scores, "평균 Acc : ", round(np.mean(scores),4))



'''
LinearSVC : 평균 Acc :  0.854
SVC : 평균 Acc :  0.6457
KNeighborsClassifier : 평균 Acc :  0.691
LogisticRegression : 평균 Acc :  0.9608
DecisionTreeClassifier : 평균 Acc :  0.9211
RandomForestClassifier : 평균 Acc :  0.9832
'''