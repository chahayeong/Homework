# 1. 데이터

import numpy as np

x_train = np.load('./_save/_npy/k59_6_train_x.npy')
x_test = np.load('./_save/_npy/k59_6_test_x.npy')
y_train = np.load('./_save/_npy/k59_6_train_y.npy')
y_test = np.load('./_save/_npy/k59_6_test_y.npy')

predict = np.load('./_save/_npy/k59_6_predict.npy')



# 2. 모델

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D

model = Sequential()
# filters = 64, kernel_size=(3,3)
model.add(Conv2D(64, (3,3), padding='same', activation='relu', input_shape=(150,150,3)))
model.add(Dense(32, activation='relu'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))


# 3. 컴파일, 훈련

from tensorflow.keras.callbacks import EarlyStopping

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_acc', patience=30, mode='min', verbose=1)

hist = model.fit(x_train, y_train, epochs=50,
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

print('acc : ', acc[-1])
print('val_acc : ', val_acc[:-1])
'''



loss = model.evaluate(x_test, y_test)
print('====================================')
print('loss : ',loss[0])
print('acc: ', loss[1])




'''
loss :  1.0999799966812134
acc:  0.4000000059604645
'''



