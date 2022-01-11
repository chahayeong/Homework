#보스턴 하우징 집값

from sklearn.datasets import load_boston
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


# 데이터 전처리
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=66, test_size=0.2)

# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)             #(101, 13) (405, 13) (101,) (405,) 

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, PowerTransformer
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)         # fit과 transfrom 같이 (train에서만)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1, 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1, 1)



# 3. 학습
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, Dropout, GlobalAveragePooling2D, MaxPool2D

model = Sequential()                      
model.add(Conv2D(filters=32, kernel_size=2,                          
                        padding='same', activation='relu', input_shape=(13, 1, 1))) 
model.add(Dropout(0.2))
model.add(Conv2D(32, 2, padding='same', activation='relu'))
# model.add(MaxPool2D())
model.add(Conv2D(64, 2, padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(64, 2, padding='same', activation='relu'))
# model.add(MaxPool2D())
model.add(Conv2D(128, 2, padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(128, 2, padding='same', activation='relu'))
# model.add(MaxPool2D())
model.add(GlobalAveragePooling2D())
model.add(Dense(1))


# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1)


import time 
start_time = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=128, verbose=2,
    validation_split=0.2, callbacks=[es])
end_time = time.time() - start_time



# 4. 예측 및 평가
loss = model.evaluate(x_test, y_test)
print("걸린시간 : ", end_time)
print('loss : ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

print("r2 score : ", r2)



'''
============== 이전 ===============
loss :  41.73356628417969
r2 score :  0.48937290503318553



걸린시간 :  13.176641464233398
loss :  12.435790061950684
r2 score :  0.8512161751944006

'''