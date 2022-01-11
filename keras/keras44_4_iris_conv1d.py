import numpy as np
from sklearn.datasets import load_iris
from sklearn.datasets import load_boston
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split        # train data, test data 분리
from sklearn.metrics import r2_score



# 1. 데이터 분석

datasets = load_iris()

x = datasets.data
y = datasets.target

print(x.shape, y.shape) # (150, 4) (150,)
print(y)



from tensorflow.keras.utils import to_categorical
y = to_categorical(y)


# train_test_split : train data와 test data 자동으로 분리 해주는 함수
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=66, \
    test_size=0.7)


from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, PowerTransformer
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)         # fit과 transfrom 같이 (train에서만)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1, 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1, 1)






# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten

model = Sequential()
model.add(Conv1D(64, 2, input_shape=(4,1)))
model.add(LSTM(64, return_sequences=True))
model.add(Conv1D(64,2))        
model.add(Flatten())            
model.add(Dense(10))
model.add(Dense(3))

model.summary()


# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)       # petience 최저점 나오면 5번까지 참다가 멈춘다 

import time 
start_time = time.time()

hist = model.fit(x_train, y_train, epochs=100, batch_size=8, \
    validation_split=0.2, callbacks=[es]) 


end_time = time.time() - start_time


print("============== 평가, 예측 ===============")
# 4. 예측 및 평가
loss = model.evaluate(x_test, y_test)
print("걸린시간 : ", end_time)
print('loss : ', loss[0])
print('accuracy : ', loss[1])


'''
============= 이전
걸린시간 :  8.093765258789062
loss :  0.14300410449504852
accuracy :  0.961904764175415

============= Conv1D
걸린시간 :  3.7071499824523926
loss :  1.0747473239898682
accuracy :  0.34285715222358704
'''