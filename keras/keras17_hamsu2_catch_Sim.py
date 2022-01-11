# 06_R2_2 카피
# 함수형으로 리폼
# 서머리로 확인



from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import numpy as np
from sklearn.metrics import r2_score 

x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])

input1 = Input(shape=(1,))
dense1 = Dense(500)(input1)
dense2 = Dense(300)(dense1)
dense3 = Dense(100)(dense2)
dense4 = Dense(50)(dense3)
dense5 = Dense(10)(dense4)
dense6 = Dense(5)(dense5)
dense7 = Dense(3)(dense6)
output1 = Dense(1)(dense7)

'''
model = Sequential()
model.add(Dense(500, input_dim=1))
model.add(Dense(300))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))
'''


model = Model(inputs=input1, outputs=output1)

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=500, batch_size=1)


model.summary()

loss = model.evaluate(x, y)
print('loss : ', loss)

y_pred = model.predict([x])
print('x의 예측값: ', y_pred)


r2 = r2_score(y, y_pred)
print("r2스코어 : ", r2)