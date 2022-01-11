from sklearn.datasets import load_boston
from tensorflow.keras.layers import Dense, LSTM, Input, Conv1D, Flatten
from tensorflow.keras.models import Sequential, Model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split        # train data, test data 분리
from sklearn.metrics import r2_score



# 1. 데이터 분석

datasets = load_boston()

x = datasets.data   
y = datasets.target


print(np.min(x), np.max(x))             # 0.0   711.0


# train_test_split : train data와 test data 자동으로 분리 해주는 함수
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=66, test_size=0.8)
                 
# print(x.shape, y.shape)         #(506, 13) (506,)     

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)




# 3. 모델 구성
model = Sequential()
model.add(Conv1D(64, 2, input_shape=(13,1)))
model.add(LSTM(64, return_sequences=True))
model.add(Conv1D(64,2))         # LSTM 다음 Conv1D가 대체적으로 성능이 좋았음
model.add(Flatten())            #해주지 않으면 3차원으로 쭉 내려감
model.add(Dense(10))
model.add(Dense(1))

model.summary()


# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

import time 
start_time = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=8)

end_time = time.time() - start_time
# 4. 예측 및 평가

y_predict = model.predict([x_test])


loss = model.evaluate(x_test, y_test)
print("걸린시간 : ", end_time)
print('loss : ', loss)

r2 = r2_score(y_test, y_predict)
print("r2 score : ", r2)



'''
====== 이전
loss :  24.82421875
r2 score :  0.6962656186872671

====== conv1D
걸린시간 :  5.829114675521851
loss :  19.496191024780273
r2 score :  0.7614561782926402
'''

