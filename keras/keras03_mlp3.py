from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#1. 데이터
x = np.array([range(10), range(21,31), range(201,211)])         
x = np.transpose(x)                 # (10, 3)

y = np.array([[1,2,3,4,5,6,7,8,9,10],[1, 1.1, 1.2, 1.3, 1.4 ,1.5 ,1.6 ,1.5 ,1.4, 1.3],[10,9,8,7,6,5,4,3,2,1]]) 
y = np.transpose(y)                 # (10, 3)



x_pred = np.array([[0, 21, 201]])
print(x_pred.shape)                 # (1, 3)



model = Sequential()
model.add(Dense(3, input_dim=3))            # 결과값 나올 개수(output), input_dim = 차원

model.compile(loss='mse', optimizer='adam')

model.fit(x, y, epochs=10, batch_size=1) 

loss = model.evaluate(x, y)
print('loss : ', loss)

result = model.predict(x_pred)
print('예측값: ', result)




y_predict = model.predict(x)

fig = plt.figure(figsize=(10, 10))
ax = fig.gca(projection='3d')

x = np.transpose(x)
y = np.transpose(y)
plt.scatter(x[0],x[1],x[2])
plt.scatter(y[0],y[1],y[2])

plt.show()
