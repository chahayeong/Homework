#keras37을 함수형으로 리폼


import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Input

# 1. 데이터
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], 
            [5,6,7], [6,7,8], [7,8,9], [8,9,10],
            [9,10,11],[10,11,12],
            [20,30,40],[30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_predict = np.array([50,60,70])

print(x.shape, y.shape)         #(13, 3) (13,)

x = x.reshape(x.shape[0], x.shape[1],1)          # (batch_size, timesteps, feature)
x_predict = x_predict.reshape(1,x_predict.shape[0],1)


# 2. 모델구성
input1 = Input(shape=(3,1)) 
model1 = GRU(32, activation='relu')(input1) 
model2 = Dense(16, activation='relu')(model1) 
model3 = Dense(8, activation='relu')(model2) 
model4 = Dense(8, activation='relu')(model3) 
model5 = Dense(4, activation='relu')(model4)
output1 = Dense(1)(model5) 

model = Model(inputs=input1, outputs=output1)


# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)

# 4. 평가, 예측

results = model.predict(x_predict)
print(results)             # [[80.13297]]