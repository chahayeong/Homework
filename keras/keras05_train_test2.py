from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])


x_train = x[:7]
y_train = y[:7]
x_test = x[7:]
y_test = y[7:]

print(x_train)
print(y_train)
print(x_test)
print(y_test)



# 2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))             # 마지막이 출력 갯수
model.add(Dense(1))             



# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x_train, y_train, epochs=1000, batch_size=1)      

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

# result = model.predict([11])
# print('11의 예측값: ', result)


y_predict = model.predict([11])
print('예측값 : ', y_predict )

