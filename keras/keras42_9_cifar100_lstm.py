from tensorflow.keras.datasets import cifar100

import numpy as np
import matplotlib.pyplot as plt
from icecream import ic

# 1. data
(x_train, y_train), (x_test, y_test) = cifar100.load_data() 

x_train = x_train.reshape(50000, 32*32*3) # (50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32*32*3) # (10000, 32, 32, 3)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = QuantileTransformer()
# scaler.fit(x_train) 
# x_train = scaler.transform(x_train) 
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) 

x_train = x_train.reshape(50000, 32*32, 3)
x_test = x_test.reshape(10000, 32*32, 3)

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
one.fit(y_train)
y_train = one.transform(y_train).toarray() # (50000, 100)
y_test = one.transform(y_test).toarray() # (10000, 100)

# 2. model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input

input1 = Input(shape=(32*32, 3))
xx = LSTM(units=20, activation='relu')(input1)
xx = Dense(256, activation='relu')(xx)
xx = Dense(128, activation='relu')(xx)
xx = Dense(100, activation='relu')(xx)
# xx = Dense(16, activation='relu')(xx)
output1 = Dense(100, activation='softmax')(xx)

model = Model(inputs=input1, outputs=output1)

# 3. comple fit // metrics 'acc'
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)

import time 
start_time = time.time()
model.fit(x_train, y_train, epochs=10, batch_size=2000, verbose=2,
    validation_split=0.05)
end_time = time.time() - start_time

# 4. predict eval -> no need to

loss = model.evaluate(x_test, y_test)
print("time : ", end_time)
print('loss : ', loss[0])
print('acc : ', loss[1])


'''
time :  413.2168846130371
loss :  4.605208396911621
acc :  0.009999999776482582
'''