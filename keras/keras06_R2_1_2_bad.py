#과제1
#1. R2를 음수가 아닌 0.5 이하로 만들어라.
#2. 데이터 건들지 않기
#3. 레이어는 인풋 아웃풋 포함 6개 이상
#4. batch_size = 1
#5. epochs는 100 이상
#6. 히든레이어의 노드는 10개 이상 1000개 이하
#7. train 70%
#--------------------------------------------------

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split    # x,y 두 값을 나누어 분리
from sklearn.metrics import r2_score                    # 결정계수 사용


# 1. 데이터
x = np.array(range(100))
y = np.array(range(1,101))

# x,y를 한꺼번에 넣음, 70%를 train으로 이용, shuffle 함
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True)



x_train = x[0:70]
y_train = y[:70]
x_test = x[-30:]
y_test = y[70:]


# 2. 모델구성
model = Sequential()
model.add(Dense(1000, input_dim=1))
model.add(Dense(500))
model.add(Dense(999))
model.add(Dense(1000))             # 마지막이 출력 갯수
model.add(Dense(999))
model.add(Dense(1))              


# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x_train, y_train, epochs=100, batch_size=1)   



# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)


y_predict = model.predict([x_test])
# print('예측값 : ', y_predict)                         


r2 = r2_score(y_test, y_predict)                    # y에 대한 원래 값과 y에대한 예측값



print("r2스코어 : ", r2)



# r2스코어 : 0.4737085244040553


