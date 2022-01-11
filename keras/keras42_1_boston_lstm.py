from sklearn.datasets import load_boston
from tensorflow.keras.layers import Dense, LSTM, Input
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
input1 = Input(shape=(13, 1))
model1 = LSTM(units=20, activation='relu')(input1)
model2 = Dense(128, activation='relu')(model1)
model3 = Dense(64, activation='relu')(model2)
model4 = Dense(32, activation='relu')(model3)
model5 = Dense(16, activation='relu')(model4)
output1 = Dense(1)(model5)

model = Model(inputs=input1, outputs=output1)


# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=8)


# 4. 예측 및 평가

y_predict = model.predict([x_test])

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

r2 = r2_score(y_test, y_predict)
print("r2 score : ", r2)



'''
loss :  24.82421875
r2 score :  0.6962656186872671
'''

