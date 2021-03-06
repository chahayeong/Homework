from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

#1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],[1, 1.1, 1.2, 1.3, 1.4 ,1.5 ,1.6 ,1.5 ,1.4, 1.3],[10,9,8,7,6,5,4,3,2,1]]) # (3,10)


print(x.shape)                      # (3, 10)
x = np.transpose(x)                 #행과 열이 바뀜
print(x.shape)                      # (10, 3)

y = np.array([[11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])
print(y.shape)                      # (1, 10)
y = np.transpose(y)    
print(y.shape)                      # (10, 1)

x_pred = np.array([[10, 1.3, 1]])
print(x_pred.shape)                 # (1, 3)





model = Sequential()
model.add(Dense(1, input_dim=3))            # 결과값 나올 개수, input_dim = 차원

model.compile(loss='mse', optimizer='adam')

model.fit(x, y, epochs=2000, batch_size=1) 

loss = model.evaluate(x, y)
print('loss : ', loss)

result = model.predict(x_pred)
print('예측값: ', result)




y_predict = model.predict(x)

x = np.transpose(x) 
plt.scatter(x[0],y)
plt.scatter(x[1],y)
plt.scatter(x[2],y)
x = np.transpose(x) 

plt.plot(x, y_predict, color='red')
plt.show()