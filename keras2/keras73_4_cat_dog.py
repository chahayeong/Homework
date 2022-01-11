from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np

# 1. 데이터
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

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, PowerTransformer
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

vgg19 = VGG19(weights='imagenet', include_top=False, input_shape=(150,150,3))
# model = VGG16()
# model = VGG19()

vgg19.trainable=True  

model = Sequential()
model.add(vgg19)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(1, activation='sigmoid'))



# 4. 컴파일, 훈련     
from tensorflow.keras.optimizers import Adam
optimizer = Adam(lr=0.1)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto', verbose=1, factor=0.5)

import time
start_time = time.time()

hist = model.fit_generator(xy_train, epochs=100, steps_per_epoch=32,
                    validation_data=xy_test,           
                    validation_steps=4, callbacks=[es, reduce_lr]) 

end_time = time.time() - start_time

# 5. 평가, 예측      

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('acc : ', acc[-1])
print('val_acc : ', val_acc[-1])


'''
acc :  1.0
val_acc :  1.0

acc :  1.0
val_acc :  1.0
'''