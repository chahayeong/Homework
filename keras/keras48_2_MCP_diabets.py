

from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
# import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from tensorflow.python.keras.saving.save import load_model
from tensorflow.python.keras.callbacks import ModelCheckpoint

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

'''
#2. 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv1D, LSTM, Flatten

model = Sequential()
model.add(Conv1D(64, 2, input_shape=(10,1)))
model.add(LSTM(64, return_sequences=True))
model.add(Conv1D(64,2))        
model.add(Flatten())           
model.add(Dense(10))
model.add(Dense(1))

model.summary()


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1)
cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto',
                    filepath='./_save/ModelCheckPoint/keras48_2_MCP.hdf5')

import time 
start_time = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=128, verbose=2,
    validation_split=0.2, callbacks=[es, cp])
end_time = time.time() - start_time

model.save('./_save/ModelCheckPoint/keras48_2_MCP.hdf5')

'''
model = load_model('./_save/ModelCheckPoint/keras48_2_MCP.hdf5')

#4. 평가, 예측
# mse, R2 사용

y_predict = model.predict([x_test])
# print("예측값 : ", y_predict)

loss = model.evaluate(x_test, y_test)
# print("걸린시간 : ", end_time)
print('loss : ', loss)

r2 = r2_score(y_test, y_predict)
print("r2 score : ", r2)



'''
================ 이전
걸린시간 :  8.930438756942749
loss :  5527.83837890625
r2 score :  0.23037180803421153

=============== Conv1D
걸린시간 :  4.891580104827881
loss :  4906.79541015625
r2 score :  0.31683826192478903


=============== MCP
loss :  4890.60546875
r2 score :  0.3190923406977574
'''