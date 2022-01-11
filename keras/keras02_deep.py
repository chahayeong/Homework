from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])

# 2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(1))             # 마지막이 출력 갯수



# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

#epochs 훈련횟수, batch_size 같이 도는 횟수 2일경우([1,2] [2,3])씩 돔
model.fit(x, y, epochs=2000, batch_size=1)      

# 4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)



result = model.predict([6])
print('4의 예측값: ', result)


"""
LOSS -  0.38775911927223206
6의 예측값 : [[5.8547473]]

"""