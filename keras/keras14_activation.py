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


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)



#2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=10, activation='relu'))                 
model.add(Dense(5, activation='relu'))                # activation='relu' 활성화함수 : 값을 퍼지지 않게 해줌
model.add(Dense(1))



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x_train, y_train, epochs=500, batch_size=1, verbose=0, validation_split=0.2)



#4. 평가, 예측
# mse, R2 사용

y_predict = model.predict(x_test)
# print("예측값 : ", y_predict)

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

r2 = r2_score(y_test, y_predict)
print("r2 score : ", r2)
