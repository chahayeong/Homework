from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])

model = Sequential()
model.add(Dense(1, input_dim=1))

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=3000, batch_size=1)

loss = model.evaluate(x, y)
print('loss : ', loss)

x_pred = model.predict([6])
print('6의 예측값: ', x_pred)


