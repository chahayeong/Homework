from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


datasets = load_diabetes()
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

# model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier()
# model = LogisticRegression()
# model = DecisionTreeClassifier()
model = RandomForestClassifier()


# 4. 예측 및 평가

scores = cross_val_score(model, x, y, cv=kfold)  
print("Acc : ", scores, "평균 Acc : ", round(np.mean(scores),4))



'''
LinearSVC : 평균 Acc :  0.0068
SVC : 평균 Acc :  0.0045
KNeighborsClassifier : 평균 Acc :  0.0
LogisticRegression : 평균 Acc :  0.0023
DecisionTreeClassifier : 평균 Acc :  0.0045
RandomForestClassifier : 평균 Acc :  0.0045
'''