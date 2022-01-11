
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import fashion_mnist


# 1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()



# 2. 전처리

x_train = x_train.reshape(60000, 28 * 28 * 1)
x_test = x_test.reshape(10000, 28 * 28 * 1)    

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, PowerTransformer
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)         
x_test = scaler.transform(x_test)

# 2차원으로 스케일링 하고 다시 4차원으로 원위치
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1) 

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)




# 3. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout
from tensorflow.python.keras.layers.core import Flatten

model = Sequential()
model.add(Conv2D(filters=128, kernel_size=(2, 2), padding='same', activation='relu' ,input_shape=(28, 28, 1))) 
model.add(Dropout(0.2))
model.add(Conv2D(128, (2,2), activation='relu'))                     
model.add(MaxPool2D())   

model.add(Conv2D(128, (2,2),padding='valid', activation='relu')) 
model.add(Dropout(0.2))  
model.add(Conv2D(128, (2,2),padding='same', activation='relu'))  
model.add(MaxPool2D()) 

model.add(Conv2D(64, (2,2), activation='relu')) 
model.add(Dropout(0.2))
model.add(Conv2D(64, (2,2),padding='same', activation='relu')) 
model.add(MaxPool2D()) 

model.add(Flatten())                                              
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))




# 4. 컴파일, 훈련       metrics=['acc']
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)

import time
start_time = time.time()

hist = model.fit(x_train, y_train, epochs=100, batch_size=128, verbose=2,
    validation_split=0.2, callbacks=[es])

end_time = time.time() - start_time



# 5. 평가, 예측      
loss = model.evaluate(x_test, y_test)
print("걸린시간: ", end_time)
print('loss : ', loss[0])
print('accuracy : ', loss[1])


'''
=================== CNN ==================
loss :  0.35646852850914
accuracy :  0.8906000256538391

=================== DNN ==================
걸린시간:  337.60853099823
loss :  0.2200598120689392
accuracy :  0.9218000173568726
'''