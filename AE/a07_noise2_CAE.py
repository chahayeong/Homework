# 06 CNN 모델로

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras import activations


# 1. 데이터
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 28, 28, 1).astype('float')/255
x_test = x_test.reshape(10000, 28, 28, 1).astype('float')/255

x_train_noised = x_train + np.random.normal(0, 0.1, size=x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape)
x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)



# 2. 모델
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, MaxPooling2D

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(hidden_layer_size, kernel_size=(2, 2), 
                input_shape=(28, 28, 1),
                activation='relu', padding='same'))
    model.add(MaxPooling2D(1,1))
    model.add(Conv2D(32, (2, 2), activation='relu', padding='same'))
    model.add(Conv2D(1, (2, 2), activation='sigmoid', padding='same'))
    return model

model = autoencoder(hidden_layer_size=154)    

model.compile(optimizer='adam', loss='mse')

model.fit(x_train, x_train, epochs=10)

output = model.predict(x_test)

from matplotlib import pyplot as plt
import random

fig, ((ax1, ax2, ax3, ax4, ax5), (ax11, ax12, ax13, ax14, ax15), (ax6, ax7, ax8, ax9, ax10)) = \
    plt.subplots(3, 5, figsize=(20, 7))     # 3 -> () () () 3개

# 이미지 5개를 무작위로 고름
random_images = random.sample(range(output.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그림
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28, 28), cmap='gray')
    if i == 0:
        ax.set_ylabel("INPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 잡음 넣은 이미지
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(x_test_noised[random_images[i]].reshape(28, 28), cmap='gray')
    if i == 0:
        ax.set_ylabel("INPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])


# 오토인코더가 출력한 이미지를 아래에 그림
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(output[random_images[i]].reshape(28, 28), cmap='gray')
    if i == 0 :
        ax.set_ylabel("OUTPUT", size=20)
        ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()
