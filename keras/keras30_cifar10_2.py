
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar10


# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()



# 2. 전처리
from sklearn.preprocessing import OneHotEncoder
one_hot = OneHotEncoder()

x_train = x_train.reshape(50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32, 32, 3)
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

y_train = one_hot.fit_transform(y_train).toarray()     
y_test = one_hot.fit_transform(y_test).toarray()         




# 3. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from tensorflow.python.keras.layers.core import Flatten

model = Sequential()
model.add(Conv2D(filters=30, kernel_size=(2, 2), padding='same' ,input_shape=(32, 32, 3))) 
model.add(Conv2D(20, (2,2), activation='relu'))                     
model.add(MaxPool2D())                                            
model.add(Conv2D(15, (2,2)))                                               
model.add(Flatten())                                              
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))




# 4. 컴파일, 훈련       metrics=['acc']
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)

model.fit(x_train, y_train, epochs=50, batch_size=8, verbose=2,
    validation_split=0.0015, callbacks=[es])




# 5. 평가, 예측      
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])


'''
loss :  1.2908146381378174
accuracy :  0.604200005531311
'''