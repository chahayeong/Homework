import numpy as np

x_train = np.load('./_save/_npy/k59_3_train_x.npy')
x_test = np.load('./_save/_npy/k59_3_test_x.npy')
y_train = np.load('./_save/_npy/k59_3_train_y.npy')
y_test = np.load('./_save/_npy/k59_3_test_y.npy')

# 2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D

model = Sequential()
model.add(Conv2D(16, (2,2), padding='same', activation='relu', input_shape=(150,150,3)))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 3. compile train
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)


hist = model.fit(x_train, y_train, epochs=50,
                callbacks=[es],
                validation_split=0.2,
                steps_per_epoch=32,
                validation_steps=4)

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

loss = model.evaluate(x_test, y_test)
print('acc : ',acc[-1])
print('loss : ',loss[0])