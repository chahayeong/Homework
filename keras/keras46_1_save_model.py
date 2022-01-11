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


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, shuffle=True, random_state=66)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델 구성
model = Sequential()
model.add(Dense(64, input_dim=10))            
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(1))

model.summary()

model.save('./_save/keras46_1_save_model_1.h5')


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)

import time
start_time = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=2, validation_split=0.2, callbacks=[es])
end_time = time.time() - start_time 
print("경과시간 : ", end_time)

model.save('./_save/keras46_1_save_model_2.h5')


#4. 평가, 예측
# mse, R2 사용

y_predict = model.predict(x_test)
# print("예측값 : ", y_predict)

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

r2 = r2_score(y_test, y_predict)
print("r2 score : ", r2)


