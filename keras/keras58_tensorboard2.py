from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,4,3,5,6,7,8,9,10])

# 2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(1))             # 마지막이 출력 갯수

from tensorflow.keras.callbacks import TensorBoard

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

tb = TensorBoard(log_dir='./_save/_graph', histogram_freq=0,
                    write_graph=True, write_images=True)

#epochs 훈련횟수, batch_size 같이 도는 횟수 2일경우([1,2] [2,3])씩 돔
model.fit(x, y, epochs=100, batch_size=1, callbacks=[tb], validation_freq=True)      

# 4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)



result = model.predict([6])
print('4의 예측값: ', result)


"""
LOSS -  0.38775911927223206
6의 예측값 : [[5.8547473]]

"""

'''
텐서보드 명령어
커맨드 창에서
cd 로 _graph 폴더로 이동
dir /w
tensorboard --logdir=.
주소 창에 복사
'''