import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras import activations


# 1. 데이터
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 784).astype('float')/255
x_test = x_test.reshape(10000, 784).astype('float')/255


# 2. 모델
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input

def autoencoder(hidden_layer_size):         # 딥하게 구성
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape=(784,),
                activation='relu'))
    model.add(Dense(units=784, activation='sigmoid'))
    return model

def autoencoder2(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape=(784,),
                activation='relu'))
    model.add(Dense(units=1024, activation='relu'))
    model.add(Dense(units=1024, activation='relu'))
    model.add(Dense(units=950, activation='relu'))
    model.add(Dense(units=900, activation='relu'))
    model.add(Dense(units=800, activation='relu'))
    model.add(Dense(units=784, activation='sigmoid'))
    return model


model1 = autoencoder(hidden_layer_size=64)
model2 = autoencoder2(hidden_layer_size=64)

model1.compile(optimizer='adam', loss='mse')
model2.compile(optimizer='adam', loss='mse')

model1.fit(x_train, x_train, epochs=10, batch_size=1024)
model2.fit(x_train, x_train, epochs=10, batch_size=1024)

output1 = model1.predict(x_test)
output2 = model2.predict(x_test)

from matplotlib import pyplot as plt
import random

fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = \
    plt.subplots(2, 5, figsize=(20, 7))



# 이미지 5개를 무작위로 고름
random_images = random.sample(range(output1.shape[0]), 5)
random_images = random.sample(range(output2.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그림
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28, 28), cmap='gray')
    if i == 0:
        ax.set_ylabel("INPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토인코더가 출력한 이미지를 아래에 그림
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(output1[random_images[i]].reshape(28, 28), cmap='gray')
    if i == 0 :
        ax.set_ylabel("OUTPUT", size=20)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_xticks([])
        ax.set_yticks([])


# 오토인코더가 출력한 이미지를 아래에 그림
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(output2[random_images[i]].reshape(28, 28), cmap='gray')
    if i == 0 :
        ax.set_ylabel("OUTPUT", size=20)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_xticks([])
        ax.set_yticks([])

plt.tight_layout()
plt.show()
