from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.metrics import r2_score 

x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])

model = Sequential()
model.add(Dense(500, input_dim=1))
model.add(Dense(300))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))



model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=10000, batch_size=1)

loss = model.evaluate(x, y)
print('loss : ', loss)

y_pred = model.predict([x])
print('x의 예측값: ', y_pred)


r2 = r2_score(y, y_pred)
print("r2스코어 : ", r2)



'''
r2스코어 :  0.8099995785432383
'''