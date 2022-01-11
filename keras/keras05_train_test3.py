from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split    # x,y 두 값을 나누어 분리


# 1. 데이터
x = np.array(range(100))
y = np.array(range(1,101))

# x,y를 한꺼번에 넣음, 70%를 train으로 이용, shuffle 함
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True)

print(x_test)  
print(y_test)    
print('--------------')

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7)

print(x_test)  
print(y_test) 
print('--------------')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.7)

print(x_test)  
print(y_test) 
print('--------------')

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, test_size=0.2)

print(x_test)  
print(y_test) 
print('--------------')

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=66)

print(x_test)  
print(y_test) 
print('--------------')




'''

x_train = x[0:70]
y_train = y[:70]
x_test = x[-30:]
y_test = y[70:]


print(x_train.shape)    # (70,)
print(y_train.shape)    # (70,)
print(x_test.shape)     # (30,)
print(y_test.shape)     # (30,)




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

'''