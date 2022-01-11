
#과제3
#보스턴 하우징 집값

from sklearn.datasets import load_boston
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split        # train data, test data 분리
from sklearn.linear_model import LinearRegression           # sklearn 선형회귀
from sklearn.metrics import r2_score



datasets = load_boston()

'''
x = datasets.data
y = datasets.target

# print(datasets)
# print(datasets.keys())            # key [data, target, feature_names, DESCR, filename]

print(x.shape)
print(y.shape)

print(datasets.feature_names)

# print(datasets.DESCR)
# print(datasets.filename)
'''


# 데이터 프레임으로 변환
df = pd.DataFrame(data=datasets.data, columns=datasets.feature_names)
df['price'] = datasets.target
# print(df)



# 1. 데이터 분석

# 상관관계 분석
correlation_matrix = df.corr().round(2)                         # 상관계수 범위 -1, 1
sns.heatmap(data=correlation_matrix, annot=True)                # 양,음 상관없이 크기가 크면 강한 상관관계
plt.show()                                                      # [PTRATIO, RM, LSTAT]

'''

# 2. x값 y값 설정

# 가격과 강한 상관관계를 가지는 feature 가져오기
x = pd.DataFrame(np.c_[df["PTRATIO"], df["RM"], df["LSTAT"]], columns=["PTRATIO", "RM", "LSTAT"])     
y = df["price"]


# train_test_split : train data와 test data 자동으로 분리 해주는 함수
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)




# 3. 학습
model = LinearRegression()                          #선형회귀 라이브러리
model.fit(x_train, y_train)




# 4. 예측 및 평가
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

print("r2 score : ", r2)
'''

'''
최대 r2 score : 0.7794479166188322
'''