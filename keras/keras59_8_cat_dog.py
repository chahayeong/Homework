# categorical_crossentropy 와 sigmoid 조립

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,        
    shear_range=0.7,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

xy_train = train_datagen.flow_from_directory(
    '../_data/cat_dog/train_set',
    target_size=(150, 150),
    batch_size=5,
    class_mode='binary',
    shuffle=True
)

xy_test = test_datagen.flow_from_directory(
    '../_data/cat_dog/test_set',
    target_size=(150, 150),
    batch_size=5,
    class_mode='binary',
    shuffle=True
)

# 2. 모델

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(32, (2, 2), input_shape=(150,150,3)))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

# 3. 컴파일, 훈련


model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['acc'])

import time
start_time = time.time()
hist = model.fit_generator(xy_train, epochs=100, steps_per_epoch=32,
                    validation_data=xy_test,           
                    validation_steps=4)    
end_time = time.time() - start_time


acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('acc : ', acc[-1])
print('val_acc : ', val_acc[-1])

print('=============================================')

print("걸린시간 : ", end_time)
# loss = model.evaluate(x_test, y_test)
print('loss : ',loss[0])
print('acc: ', loss[1])
