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


'''

y = y.reshape(-1,1)                 # (3,) 1차원을 2차원으로

x = np.transpose(x) 
y = np.transpose(y)

print(x.shape, y.shape)             # (442, 10) (442,) -> train test 분리, validation split

print(datasets.feature_names)       # ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
print(datasets.DESCR)

print(y[:30])
print(np.min(y), np.max(y))
'''


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, shuffle=True, random_state=66)



#2. 모델 구성
model = Sequential()
model.add(Dense(200, input_dim=10))                 # 첫 오류 : input_dim 잘 확인하기.
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(80))
model.add(Dense(60))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(1))



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x_train, y_train, epochs=1200, batch_size=1, verbose=2, validation_split=0.2)



#4. 평가, 예측
# mse, R2 사용

y_predict = model.predict(x_test)
# print("예측값 : ", y_predict)

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

r2 = r2_score(y_test, y_predict)
print("r2 score : ", r2)


# 과제1. 0.62이상 올려서 github upload
# r2 score :  0.5271855226980122
