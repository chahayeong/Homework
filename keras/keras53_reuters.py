from tensorflow.keras.datasets import reuters
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words=10000, test_split=0.2
)

print(x_train[0], type(x_train[0]))
# print(x_train[1], type(x_train[1]))
print(y_train[0])                   # 3

print(len(x_train[0]), len(x_train[1]))     # 87, 56

# print(x_train[0].shape)

print(x_train.shape, x_test.shape)  # (8982,) (2246,)
print(y_train.shape, y_test.shape)  # (8982,) (2246,)

print(type(x_train))                # <class 'numpy.ndarray'>

print("뉴스기사의 최대길이 : ", max(len(i) for i in x_train))   # 2376
print("뉴스기사의 평균길이 : ", sum(map(len, x_train)) / len(x_train))  # 145.5

# plt.hist([len(s) for s in x_train], bins=50)
# plt.show()

# 전처리
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

x_train = pad_sequences(x_train, maxlen=100, padding='pre')
x_test = pad_sequences(x_test, maxlen=100, padding='pre')
print(x_train.shape, x_test.shape)
print(type(x_train), type(x_train[0]))  # <class 'numpy.ndarray'> <class 'numpy.ndarray'>
print(x_train[1])

# y 확인
print(np.unique(y_train))

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape, y_test.shape)

# 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, GRU

model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=10, input_length=230))
model.add(GRU(32, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(46, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam'
                , metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=40, mode='min', verbose=1)

import time

start_time = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=4000, verbose=2,
    validation_split=0.175, callbacks=[es])
end_time = time.time() - start_time

# 4. 예측, 평가

loss = model.evaluate(x_test, y_test, batch_size=1024)
print("time : ", end_time)
print('loss : ', loss[0])
print('acc : ', loss[1])


'''
time :  55.494786739349365
loss :  2.0563406944274902
acc :  0.47951915860176086
'''