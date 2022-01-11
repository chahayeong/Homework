'''
실습2
본인 사진으로 predict하기 d:\data 안에 사진 넣고
내가 남자%인지 여자%인지?
acc는 몇인지?
'''


# 1. 데이터

import numpy as np

x_train = np.load('./_save/_npy/k59_5_train_x.npy')
x_test = np.load('./_save/_npy/k59_5_test_x.npy')
y_train = np.load('./_save/_npy/k59_5_train_y.npy')
y_test = np.load('./_save/_npy/k59_5_test_y.npy')

predict = np.load('./_save/_npy/k59_5_predict.npy')


# 2. 모델

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D

model = Sequential()
model.add(Conv2D(16, (2,2), padding='same', activation='relu', input_shape=(150,150,3)))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# 3. 컴파일, 훈련

from tensorflow.keras.callbacks import EarlyStopping

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)

hist = model.fit(x_train, y_train, epochs=100,
                callbacks=[es],
                validation_split=0.2,
                steps_per_epoch=32,
                validation_steps=4)


# 4. 예측, 평가
'''
acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('=======================================')
print('acc: ', acc[-1])
print('loss: ', loss[0])
print('val acc: ', val_acc)
print('val loss: ', val_loss)
print('======================================')
'''


loss = model.evaluate(x_test, y_test)
print('====================================')
print('loss : ',loss[0])
print('acc: ', loss[1])
print('====================================')

y_predict = model.predict(predict)
percentage1 = y_predict * 100
percentage2 = (1 - y_predict) * 100
print('여자일 확률: ', percentage1)
print('남자일 확률: ', percentage2)
print(percentage1 + percentage2)

                           

'''
====================================
loss :  2.1611788272857666
acc:  0.800000011920929
====================================
여자일 확률:  [[99.014824]]
남자일 확률:  [[0.9851754]]
[[100.]]
'''