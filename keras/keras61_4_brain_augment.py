# 훈련데이터를 기존 데이터 20% 더할것
# 성과비교
# save_dir도 temp에 넣을 것
# 증폭데이터는 temp에 저장후 훈련 끝난 뒤 삭제


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
    '../_data/brain/train',
    target_size=(150, 150),
    batch_size=200,
    class_mode='binary',
    shuffle=True
)

xy_test = test_datagen.flow_from_directory(
    '../_data/brain/test',
    target_size=(150, 150),
    batch_size=200,
    class_mode='binary',
    shuffle=True
)

x_train = xy_train[0][0] 
y_train = xy_train[0][1] 

x_test = xy_test[0][0] 
y_test = xy_test[0][1] 
augment_size = 160

randidx = np.random.randint(x_train.shape[0], size=augment_size) 

x_argmented = x_train[randidx].copy()
y_argmented = y_train[randidx].copy()

x_argmented = x_argmented.reshape(x_argmented.shape[0], 150, 150, 3) 
x_train = x_train.reshape(x_train.shape[0], 150, 150, 3) 
x_test = x_test.reshape(x_test.shape[0], 150, 150, 3) 

x_argmented = train_datagen.flow(x_argmented, 
                                np.zeros(augment_size),
                                batch_size=augment_size,
                                shuffle=False).next()[0]

x_train = np.concatenate((x_train, x_argmented)) 
y_train = np.concatenate((y_train, y_argmented)) 

# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D

model = Sequential()
model.add(Conv2D(16, (2,2), padding='same', activation='relu', input_shape=(150,150,3)))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 3. 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_acc', patience=10, mode='auto', verbose=1)

import time 

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=100, verbose=2,
    validation_split=0.2, callbacks=[es], steps_per_epoch=32,
                validation_steps=4)
end_time = time.time() - start_time

# 4. 평가

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

loss = model.evaluate(x_test, y_test)
print('acc : ',acc[-1])
print('val_acc : ',val_acc[-1])
print('val_loss : ',val_loss[-1])   


'''
acc :  0.85546875
val_acc :  0.53125
val_loss :  0.7091835141181946
'''