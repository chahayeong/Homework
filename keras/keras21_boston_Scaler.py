#보스턴 하우징 집값

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


print(np.min(x), np.max(x))             # 0.0   711.0

# 데이터 전처리
# x = x/711.
# x = x/np.max(x)
# x = (x - np.min(x)) / (np.max(x) - np.min(x))

# train_test_split : train data와 test data 자동으로 분리 해주는 함수
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=66, test_size=0.8)


# print(x.shape)
# print(x_train.shape)
# print(x_test.shape)

from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# scaler = MinMaxScaler()                           
# scaler = StandardScaler()

# scaler = MaxAbsScaler()
# scaler = RobustScaler()
# scaler = QuantileTransformer()
scaler = PowerTransformer()

scaler.fit(x_train)                                
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)                    






# 3. 학습
model = Sequential()                      
model.add(Dense(128, activation='relu', input_shape=(13,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))


# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=8)


# 4. 예측 및 평가
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

print("r2 score : ", r2)



'''
MaxAbsScaler
loss :  24.08354949951172
r2 score :  0.7053280060227751


RobustScaler
loss :  18.33221435546875
r2 score :  0.7756979203515335


QuantileTransformer
loss :  27.194902420043945
r2 score :  0.667259377465375


PowerTransformer
loss :  36.897132873535156
r2 score :  0.5485485880810479

'''