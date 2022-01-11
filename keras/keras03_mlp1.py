from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

#1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
[1, 1.1, 1.2, 1.3, 1.4 ,1.5 ,1.6 ,1.5 ,1.4, 1.3]])

print(x.shape)                      # (2, 10)
x = np.transpose(x)                 #행과 열이 바뀜
print(x.shape)                      #(10, 2)

y = np.array([[11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])
print(y.shape)                      # (1, 10)
y = np.transpose(y)    
print(y.shape)                      # (10, 1)

x_pred = np.array([[10, 1.3]])
print(x_pred.shape)                 # (1, 2)




model = Sequential()
model.add(Dense(1, input_dim=2))

model.compile(loss='mse', optimizer='adam')

model.fit(x, y, epochs=100, batch_size=1) 

loss = model.evaluate(x, y)
print('loss : ', loss)

result = model.predict(x_pred)
print('예측값: ', result)



#####

y_predict = model.predict(x)

x = np.transpose(x)

plt.scatter(x[0], y)
plt.scatter(x[1], y)

x = np.transpose(x)

plt.plot(x, y_predict, color='red')
plt.show()








"""
1. [1, 2, 3]
2. [[1, 2, 3]]                  - 1행 3렬
3. [[1, 2],[3, 4], [5, 6]]      - 3행 2열
4. [[[1, 2, 3], [4, 5, 6]]]     1, 2, 3
5. [[1, 2], [3, 4], [5, 6]]     1, 3, 2
6. [[[1],[2]], [[3],[4]]]       2, 2, 1


피처=컬럼=열= 특성
열 우선, 행무시
"""