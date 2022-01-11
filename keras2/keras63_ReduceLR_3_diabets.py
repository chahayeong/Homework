

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

print(x.shape, y.shape)         # (442, 10) (442,)



x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, shuffle=True, random_state=66)

from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)                           
                 

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)


#2. 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, LSTM

input1 = Input(shape=(10, 1))
xx = LSTM(units=20, activation='relu')(input1)
xx = Dense(128, activation='relu')(xx)
xx = Dense(64, activation='relu')(xx)
xx = Dense(32, activation='relu')(xx)
xx = Dense(16, activation='relu')(xx)
output1 = Dense(1)(xx)

model = Model(inputs=input1, outputs=output1)


#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam
optimizer = Adam(lr=0.1)

model.compile(loss='mse', optimizer=optimizer, metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto', verbose=1, factor=0.5)

import time 
start_time = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=128, verbose=2,
    validation_split=0.2, callbacks=[es, reduce_lr])
end_time = time.time() - start_time



#4. 평가, 예측
# mse, R2 사용

y_predict = model.predict([x_test])
# print("예측값 : ", y_predict)

loss = model.evaluate(x_test, y_test)
print("걸린시간 : ", end_time)
print('loss : ', loss)

r2 = r2_score(y_test, y_predict)
print("r2 score : ", r2)



'''
걸린시간 :  8.930438756942749
loss :  5527.83837890625
r2 score :  0.23037180803421153
'''


''' optimizer
걸린시간 :  8.830136775970459
loss :  [4725.18017578125, 0.0]
r2 score :  0.3421241010737075
'''