import numpy as np
from sklearn.datasets import load_wine


# 1. 데이터 분석

datasets = load_wine()

x = datasets.data
y = datasets.target

# print(x.shape, y.shape) # (178, 13) (178,)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=66, \
    test_size=0.7)


from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()
scaler.fit(x_train)                                
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) 


# 2. 모델 구성


from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

allAlgorithms = all_estimators(type_filter='classifier')

for ( name , algorithm ) in allAlgorithms:
    try :
        model = algorithm()

        model.fit(x_train, y_train)

        y_predict = model.predict(x_test)
        acc = accuracy_score(y_test, y_predict)
        print(name, '의 정답률 : ', acc)
    except:
        print(name, '는 없음')
        continue


