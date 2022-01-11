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
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, Dropout, GlobalAveragePooling2D, MaxPool2D, LSTM

input = Input(shape=(4, 1))
model1 = LSTM(32, activation='relu')(input)
model2 = Dense(64, activation='relu')(model1)
model3 = Dense(64, activation='relu')(model2)
model4 = Dense(8, activation='relu')(model3)
output = Dense(3, activation='sigmoid')(model4)

model = Model(inputs=input, outputs=output)



# 3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam
optimizer = Adam(lr=0.1)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)       # petience 최저점 나오면 5번까지 참다가 멈춘다 
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto', verbose=1, factor=0.5)

import time 
start_time = time.time()

hist = model.fit(x_train, y_train, epochs=100, batch_size=8, \
    validation_split=0.2, callbacks=[es, reduce_lr]) 


end_time = time.time() - start_time


print("============== 평가, 예측 ===============")
# 4. 예측 및 평가
loss = model.evaluate(x_test, y_test)
print("걸린시간 : ", end_time)
print('loss : ', loss[0])
print('accuracy : ', loss[1])


'''
걸린시간 :  8.093765258789062
loss :  0.14300410449504852
accuracy :  0.961904764175415
'''

''' optimizer
걸린시간 :  3.1707019805908203
loss :  1.1013725996017456
accuracy :  0.3333333432674408
'''