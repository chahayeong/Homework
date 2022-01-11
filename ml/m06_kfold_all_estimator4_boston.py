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


from sklearn.utils import all_estimators

from sklearn.model_selection import KFold, cross_val_score

allAlgorithms = all_estimators(type_filter='regressor')         # 보스턴은 regressor, 다른 예제는 classifier 사용

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

kfold = KFold(n_splits=5, shuffle=True, random_state=66)

for ( name , algorithm ) in allAlgorithms:
    try :
        model = algorithm()

        scores = cross_val_score(model, x, y, cv=kfold)
        print(name, scores, "평균 Acc: ", round(np.mean(scores), 4))

    except:
        print(name, '없음')