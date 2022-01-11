
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape, y_train.shape)           # (60000, 28, 28) (60000,)  흑백데이터라 3차원
# print(x_test.shape, y_test.shape)             # (10000, 28, 28) (10000,)


x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)           

# print(np.unique(y_train))                       # [0 1 2 3 4 5 6 7 8 9]



# 3. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from tensorflow.python.keras.layers.core import Flatten

model = Sequential()
model.add(Conv2D(filters=100, kernel_size=(2, 2), padding='same' ,input_shape=(28, 28, 1))) 
model.add(Conv2D(20, (2,2), activation='relu'))                     
model.add(MaxPool2D())                                            
model.add(Conv2D(15, (2,2)))                                               
model.add(Flatten())                                              
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))



# 4. 컴파일, 훈련       metrics=['acc']
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
es = EarlyStopping(monitor='val_loss', patience=7, mode='min', verbose=1)
tb = TensorBoard(log_dir='./_save/_graph', histogram_freq=0,
                    write_graph=True, write_images=True)

import time
start_time = time.time()
model.fit(x_train, y_train, epochs=20, batch_size=256, verbose=1,
    validation_split=0.25, callbacks=[es, tb])
end_time = time.time() - start_time


# 5. 평가, 예측         predict 할 필요 없음 (acc로만 판단)
loss = model.evaluate(x_test, y_test)
print("걸린시간: ", end_time)
print('loss : ', loss[0])
print('accuracy : ', loss[1])



'''
loss :  0.08869405835866928
accuracy :  0.9807999730110168
'''




