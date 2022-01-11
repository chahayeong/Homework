import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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

xy_train = train_dataagen.flow_from_directory(
    '../_data/brain/train',
    target_size=(150, 150),
    batch_size=5,
    class_mode='binary',
    shuffle=True        
)
 # 3개이상이면 categorical
 # Found 120 images belonging to 2 classes.

# print(xy_train)
# <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x000001B3AE618550>
# print(xy_train[0])
# print(xy_train[0][0])     # x값
# print(xy_train[0][1])     # y값
# print(xy_train[0][2])     # 없음
print(xy_train[0][0].shape, xy_train[0][1].shape) # (5, 150, 150, 3) (5,)
                                                # batch?_size 6으로 변경시 (6, 150, 150, 3) (6,)


# print(xy_train[31][1])      # 마지막 배치 y
# print(xy_train[32][1])      # 없음

# print(type(xy_train))           # <class 'tensorflow.python.keras.preprocessing.image.DirectoryIterator'>
# print(type(xy_train[0]))        # <class 'tuple'>
# print(type(xy_train[0][0]))     # <class 'numpy.ndarray'>
# print(type(xy_train[0][1]))     # <class 'numpy.ndarray'>



