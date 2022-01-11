import numpy as np
from sklearn.model_selection import train_test_split 
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.metrics import r2_score
from sklearn.datasets import load_breast_cancer


# 이진분류 모델


datasets = load_breast_cancer()

# print(datasets.DESCR)
# print(datasets.feature_names)


# 1. 데이터 분석
x = datasets.data
y = datasets.target

# print(x.shape, y.shape)             # (569, 30) (569,)

# print(y[:20])
# print(np.unique(y))                 # 특이한 부분 있는지 (종류 출력)


x_train, x_test, y_train, y_test = train_test_split(x,y, shuffle=True, \
    random_state=66, test_size=0.7)

# print(x.shape)                        # (569, 30)
# print(x_train.shape)                  # (113, 30)
# print(x_test.shape)                   # (456, 30)

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

from sklearn.metrics import accuracy_score
from sklearn.utils import all_estimators
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


'''
# 3. 컴파일 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측

from sklearn.metrics import r2_score, accuracy_score

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print("accuracy_score : ", acc)
'''