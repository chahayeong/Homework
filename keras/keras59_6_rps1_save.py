import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# 1. 데이터
train_dataagen = ImageDataGenerator(
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

# class mode = 'categorical'
xy_train = train_dataagen.flow_from_directory(
    '../_data/rps',
    target_size=(150, 150),
    batch_size=5,
    class_mode='categorical',
    shuffle=True    
)

xy_test = test_datagen.flow_from_directory(
    '../_data/rps',
    target_size=(150,150),
    batch_size=5,
    class_mode='categorical'
)

predict = test_datagen.flow_from_directory(
    '../_data/hihyrps',
    target_size=(150,150),
    batch_size=5,
    class_mode='categorical'
)

np.save('./_save/_npy/k59_6_train_x.npy', arr=xy_train[0][0])
np.save('./_save/_npy/k59_6_train_y.npy', arr=xy_train[0][1])
np.save('./_save/_npy/k59_6_test_x.npy', arr=xy_test[0][0])
np.save('./_save/_npy/k59_6_test_y.npy', arr=xy_test[0][1])

np.save('./_save/_npy/k59_6_predict.npy', arr=predict[0][0])
