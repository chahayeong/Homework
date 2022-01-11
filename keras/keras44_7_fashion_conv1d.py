import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist

# 1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data() # (60000, 28, 28) (60000,) (10000, 28, 28) (10000,)

x_train = x_train.reshape(60000, 28*28) # (60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28*28) # (10000, 28, 28, 1)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = QuantileTransformer()
# scaler.fit(x_train) 
# x_train = scaler.transform(x_train) 
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) 

x_train = x_train.reshape(60000, 28, 28) # (60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28) # (10000, 28, 28, 1)

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
one.fit(y_train)
y_train = one.transform(y_train).toarray() # (60000, 10)
y_test = one.transform(y_test).toarray() # (10000, 10)

# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPool1D, GlobalAveragePooling1D, Dropout

model = Sequential()
model.add(Conv1D(filters=32, kernel_size=2, padding='same',                        
                        activation='relu' ,input_shape=(28, 28))) 
model.add(Conv1D(32, 2, padding='same', activation='relu'))                   
model.add(MaxPool1D())                                         
model.add(Conv1D(64, 2, padding='same', activation='relu'))                   
model.add(Conv1D(64, 2, padding='same', activation='relu'))    
model.add(Flatten())                                              
model.add(Dense(256, activation='relu'))
model.add(Dense(124, activation='relu'))
model.add(Dense(84, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=15, mode='min', verbose=1)

import time 

start_time = time.time()
model.fit(x_train, y_train, epochs=50, batch_size=8, verbose=2,
    validation_split=0.2, callbacks=[es])
end_time = time.time() - start_time

# 4. 평가

loss = model.evaluate(x_test, y_test)
print("걸린시간 : ", end_time)
print('loss : ', loss[0])
print('acc: ', loss[1])


'''
========== 이전
loss :  1.658949375152588
accuracy :  0.3084000051021576

=================
걸린시간 :  378.0792021751404
loss :  0.5566650032997131
acc:  0.8845999836921692

'''