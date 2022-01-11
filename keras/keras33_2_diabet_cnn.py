

from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
# import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


datasets = load_diabetes()

x = datasets.data
y = datasets.target

# print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, shuffle=True, random_state=66)

from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)                           
                 

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1, 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1, 1)


#2. 모델 구성
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



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1)

import time 
start_time = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=128, verbose=2,
    validation_split=0.2, callbacks=[es])
end_time = time.time() - start_time



#4. 평가, 예측
# mse, R2 사용

y_predict = model.predict(x_test)
# print("예측값 : ", y_predict)

loss = model.evaluate(x_test, y_test)
print("걸린시간 : ", end_time)
print('loss : ', loss)

r2 = r2_score(y_test, y_predict)
print("r2 score : ", r2)



'''
================= 이전 =================
loss :  6904.12353515625
r2 score :  0.03875484898340287

======= Conv2D =======
걸린시간 :  14.455637693405151
loss :  5416.2548828125
r2 score :  0.24590736217576992
'''