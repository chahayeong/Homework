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

from sklearn.utils import all_estimators

from sklearn.model_selection import KFold, cross_val_score




# 1. 데이터 분석

datasets = load_iris()

x = datasets.data
y = datasets.target





# 2. 모델 구성

# allAlgorithms = all_estimators(type_filter='classifier')
allAlgorithms = all_estimators(type_filter='classifier')

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







