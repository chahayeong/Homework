# 실습 diabets
# 1. ross와 r2로 평가
# MinMax와 Standard 결과를 명시
#

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


# caler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)                                
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)                   



#2. 모델 구성
model = Sequential()
model.add(Dense(220,activation='relu', input_dim=10))                 
model.add(Dense(100,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(50 ,activation='relu'))
model.add(Dense(1))



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x_train, y_train, epochs=100, batch_size=8, verbose=2, validation_split=0.2)



#4. 평가, 예측
# mse, R2 사용

y_predict = model.predict(x_test)
# print("예측값 : ", y_predict)

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

r2 = r2_score(y_test, y_predict)
print("r2 score : ", r2)



'''
MinMaxScaler
loss :  4003.37744140625
r2 score :  0.4426190242139002


StandardScaler
loss :  5326.08837890625
r2 score :  0.2584611297988286
'''