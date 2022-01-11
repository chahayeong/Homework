# 모델 완성 후 비교 (loss, val_loss, acc, val_acc)
# fathion mnist와 결과 비교

from tensorflow.keras.datasets import fashion_mnist
import numpy as np

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=False,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=10,
    zoom_range=0.10,         
    shear_range=0.5,
    fill_mode='nearest'
)

augment_size=40000

randidx = np.random.randint(x_train.shape[0], size=augment_size)
print(x_train.shape[0])     # 60000
print(randidx)              
print(randidx.shape)        # (40000,)

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

# print(x_augmented.shape)            # (40000, 28, 28, 1) x_augmented[0] = 40000

x_augmented = x_augmented.reshape(x_augmented.shape[0], 28, 28, 1)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_augmented = train_datagen.flow(x_augmented, np.zeros(augment_size),
                                batch_size=augment_size, shuffle=False).next()[0]

print(x_augmented.shape)        # (40000, 28, 28, 1)

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))

print(x_train.shape, y_train.shape) # (100000, 28, 28, 1) (100000,)


# 모델

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D

model = Sequential()
model.add(Conv2D(64, (2,2), padding='same', activation='relu', input_shape=(28,28,1)))
model.add(Dense(32, activation='relu'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='softmax'))



# 훈련

from tensorflow.keras.callbacks import EarlyStopping

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)

hist = model.fit(x_train, y_train, epochs=300,
                callbacks=[es],
                validation_split=0.2,
                steps_per_epoch=32,
                validation_steps=4)


# 평가

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

loss = model.evaluate(x_test, y_test)
print('====================================')
print('acc : ', acc[-1])
print('val_acc : ', val_acc[-1])

print('loss : ',loss[0])
print('acc: ', loss[1])


'''
acc :  0.10055000334978104
val_acc :  0.10395000129938126
loss :  -3678019780608.0
acc:  0.10000000149011612

acc :  0.09969999641180038
val_acc :  0.10254999995231628
loss :  -840962985164800.0
acc:  0.10000000149011612
'''