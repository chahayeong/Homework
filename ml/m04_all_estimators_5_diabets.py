

from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
# import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


datasets = load_diabetes()
x = datasets.data
y = datasets.target

# print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, shuffle=True, random_state=66)

from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer

scaler = RobustScaler()


scaler.fit(x_train)                                
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)                   



#2. 모델 구성

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
