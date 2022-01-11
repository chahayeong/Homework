import numpy as np
# import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential, Model
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


datasets = load_diabetes()
x = datasets.data
y = datasets.target


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, shuffle=True, random_state=66)



#2. 모델 구성

input1 = Input(shape=(10,))
dense1 = Dense(200)(input1)
dense2 = Dense(100)(dense1)
dense3 = Dense(100)(dense2)
dense4 = Dense(80)(dense3)
dense5 = Dense(60)(dense4)
dense6 = Dense(30)(dense5)
dense7 = Dense(30)(dense6)
output1 = Dense(1)(dense7)

'''
model = Sequential()
model.add(Dense(200, input_dim=10))                 # 첫 오류 : input_dim 잘 확인하기.
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(80))
model.add(Dense(60))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(1))
'''


#3. 컴파일, 훈련

model = Model(inputs=input1, outputs=output1)
model.summary()

model.compile(loss='mse', optimizer='adam')

model.fit(x_train, y_train, epochs=1000, batch_size=1, verbose=2, validation_split=0.2)



#4. 평가, 예측
# mse, R2 사용

y_predict = model.predict(x_test)
# print("예측값 : ", y_predict)

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

r2 = r2_score(y_test, y_predict)
print("r2 score : ", r2)