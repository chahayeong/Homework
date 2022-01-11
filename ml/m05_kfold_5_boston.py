from sklearn.datasets import load_boston
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split        # train data, test data 분리
from sklearn.metrics import r2_score



# 1. 데이터 분석

datasets = load_boston()

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
LinearSVC : 
SVC : 
KNeighborsClassifier : 
LogisticRegression : 
DecisionTreeClassifier :
RandomForestClassifier : 
'''