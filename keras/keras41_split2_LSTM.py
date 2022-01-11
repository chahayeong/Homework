# 1~100까지 데이터를
'''
      x              y
1, 2, 3, 4, 5        6
...
95,96,97,98,99      100
'''

# 96, 97, 98, 99, 100       ?
# 101, 102, 013, 014, 105   ?
# 예상 결과값 : 101 102 103 104 105 106
# 평가지표 R2, RMSE


import numpy as np
x_data = np.array(range(1, 101))
x_predict = np.array(range(96, 105))



size = 6

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

dataset = split_x(x_data, size)

# print(dataset)

x = dataset[:, :5]
y = dataset[:, 5]

# print("x : \n", x)
# print("y : ", y)

# print(x.shape, y.shape)         # (95, 5) (95,)

x = x.reshape(x.shape[0], x.shape[1], 1)
x_predict = x_predict.reshape(1,x_predict.shape[0],1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True)

# 모델 구성
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Input

input = Input(shape=(5,1))
model1 = LSTM(100, activation='relu')(input)
model2 = Dense(82, activation='relu')(model1)
model3 = Dense(64, activation='relu')(model2)
model4 = Dense(32, activation='relu')(model3)
model5 = Dense(16, activation='relu')(model4)
output = Dense(1)(model5)

model = Model(inputs=input, outputs=output)


# 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)

# 평가, 예측
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

y_predict = model.predict([x_test])
r2 = r2_score(y_test, y_predict)
print("r2스코어 : ", r2)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("rmse score : ", rmse)


'''
r2스코어 :  0.9999970061727377
rmse score :  0.04737121431936013
'''