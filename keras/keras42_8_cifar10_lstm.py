from tensorflow.keras.datasets import cifar10

import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data() 

x_train = x_train.reshape(50000, 32*32*3)
x_test = x_test.reshape(10000, 32*32*3)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = QuantileTransformer()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) 

x_train = x_train.reshape(50000, 32*32, 3)
x_test = x_test.reshape(10000, 32*32, 3)

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
one.fit(y_train)
y_train = one.transform(y_train).toarray() # (50000, 10)
y_test = one.transform(y_test).toarray() # (10000, 10)


# 2. 모델
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input

input1 = Input(shape=(32*32, 3))
xx = LSTM(units=10, activation='relu')(input1)
# xx = Dense(128, activation='relu')(xx)
# xx = Dense(64, activation='relu')(xx)
# xx = Dense(32, activation='relu')(xx)
xx = Dense(16, activation='relu')(xx)
output1 = Dense(10, activation='softmax')(xx)

model = Model(inputs=input1, outputs=output1)

# 3. 컴파일, 훈련 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)

import time 

start_time = time.time()
model.fit(x_train, y_train, epochs=10, batch_size=518, verbose=1,
    validation_split=0.2, callbacks=[es])
end_time = time.time() - start_time

# 4. predict eval -> no need to

loss = model.evaluate(x_test, y_test)
print("time : ", end_time)
print('loss : ', loss[0])
print('acc : ', loss[1])

'''
time :  924.4618992805481
loss :  2.3025963306427
acc :  0.10000000149011612
'''