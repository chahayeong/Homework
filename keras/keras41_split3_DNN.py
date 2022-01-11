# 1~100까지 데이터를
'''
      x              y
1, 2, 3, 4, 5        6
...
95,96,97,98,99      100
'''

# 96, 97, 98, 99, 100       ?
# 101, 102, 013, 014, 105   ?
# 예상 결과값 : 101 102 103 104 105 106
# 평가지표 R2, RMSE
# R2와 RMS3 비교

import numpy as np

x_data = np.array(range(1, 101))
x_pred = np.array(range(96, 106))

size1 = 6
size2 = 5

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

dataset = split_x(x_data, size1)

x_pred = split_x(x_pred, size2) # (6, 5)

x = dataset[:, :-1] # (95, 5)  
y = dataset[:, -1] # (95,)

# print(x.shape, y.shape, x_pred.shape)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y,
      test_size=0.2, shuffle=True, random_state=66)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, GlobalAveragePooling2D, Dropout

model = Sequential()
model.add(Dense(270, activation='relu', input_shape=(5,)))
model.add(Dense(240, activation="relu"))
model.add(Dense(200, activation="relu"))
model.add(Dense(124, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

# 3. 컴파일

model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=20, mode='min', verbose=1)
import time 

start_time = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=64,
        validation_split=0.1, callbacks=[es])
end_time = time.time() - start_time

# 4. 예측 및 평가
from sklearn.metrics import r2_score, mean_squared_error
y_pred = model.predict(x_test)
print("time : ", end_time)
# print('y_pred : \n', y_pred) 

r2 = r2_score(y_test, y_pred)
print('r2 스코어 : ', r2)

def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))
rmse = RMSE(y_test, y_pred)
print('rmse score : ', rmse)





'''
======== 이전
r2스코어 :  0.9999970061727377
rmse score :  0.04737121431936013

======== 
r2스코어 :  0.9999834269806598
rmse score :  0.1013692807379422
'''