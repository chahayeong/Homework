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

# x_augmented 에서 x는 0번째, y는 1번째이므로 next()[0]으로 x만 빼줌

print(x_augmented.shape)        # (40000, 28, 28, 1)

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))

print(x_train.shape, y_train.shape) # (100000, 28, 28, 1) (100000,)
