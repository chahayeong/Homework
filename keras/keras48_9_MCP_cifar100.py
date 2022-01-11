from tensorflow.keras.datasets import cifar100

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.saving.save import load_model

# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data() 

x_train = x_train.reshape(50000, 32*32*3) # (50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32*32*3) # (10000, 32, 32, 3)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = StandardScaler()
# scaler.fit(x_train) 
# x_train = scaler.transform(x_train) 
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) 

x_train = x_train.reshape(50000, 32*32, 3) # (50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32*32, 3) # (10000, 32, 32, 3)

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
one.fit(y_train)
y_train = one.transform(y_train).toarray() # (50000, 100)
y_test = one.transform(y_test).toarray() # (10000, 100)


'''
# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPool1D, GlobalAveragePooling1D, Dropout

model = Sequential()
model.add(Conv1D(filters=32, kernel_size=2, padding='same',                        
                        activation='relu' ,input_shape=(32*32, 3))) 
model.add(Conv1D(32, 2, padding='same', activation='relu'))                   
model.add(MaxPool1D())                                         
model.add(Conv1D(64, 2, padding='same', activation='relu'))                   
model.add(Conv1D(64, 2, padding='same', activation='relu'))    
model.add(Flatten())                                              
model.add(Dense(256, activation='relu'))
model.add(Dense(124, activation='relu'))
# model.add(Dense(84, activation='relu'))
model.add(Dense(100, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)
cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto',
                    filepath='./_save/ModelCheckPoint/keras48_9_MCP.hdf5')

import time 
start_time = time.time()
model.fit(x_train, y_train, epochs=10, batch_size=8, verbose=2,
    validation_split=0.05, callbacks=[es])
end_time = time.time() - start_time

model.save('./_save/ModelCheckPoint/keras48_9_MCP.hdf5')

'''

model = load_model('./_save/ModelCheckPoint/keras48_9_MCP.hdf5')

# 4. 평가

loss = model.evaluate(x_test, y_test)
# print("걸린시간 : ", end_time)
print('loss : ', loss[0])
print('acc : ', loss[1])


'''
================= 이전
time :  413.2168846130371
loss :  4.605208396911621
acc :  0.009999999776482582

================= conv1D
걸린시간 :  340.2104697227478
loss :  7.958601474761963
acc :  0.24089999496936798

============= MCP
loss :  8.245804786682129
acc :  0.243599995970726
'''