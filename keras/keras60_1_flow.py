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

# 1. ImageDataGenerator 정의
# 2. 파일에서 땡겨오려면 -> flow_from_directory()  // x,y 가 튜플형태로 뭉침 xy_train
# 3. 데이터에에서 땡겨오려면 -> flow()  // x, y로 나뉨

augument_size=100
x_data = train_datagen.flow(
    np.tile(x_train[0].reshape(28*28), augument_size).reshape(-1, 28, 28, 1),   # x
    np.zeros(augument_size),        # y
    batch_size=augument_size,
    shuffle=False
).next()   

# .next = Iterator 방식으로 변환

'''
np.tile 배열을 반목하면서 새로운 축 추가
( 반목할 배열, 반복 횟수 )
반복횟수는 숫자나 배열도 가능
2 -> 같은 차원으로 2번 반복
(2,2) -> 2차원으로 2번 반복
이미지 참고
'''
 

print(type(x_data))             # <class 'tensorflow.python.keras.preprocessing.image.NumpyArrayIterator'>
                                # .next() -> <class 'tuple'>
                               
print(type(x_data[0]))          # <class 'tuple'>
                                # .next() -> <class 'numpy.ndarray'>

print(type(x_data[0][0]))       # <class 'numpy.ndarray'>
                                # .next() -> <class 'numpy.ndarray'>

print(x_data[0][0].shape)       # (100, 28, 28, 1)
                                # .next() -> (28, 28, 1)

print(x_data[0][1].shape)       # (100,)
                                # .next() -> (28, 28, 1)


 
import matplotlib.pyplot as plt
plt.figure(figsize=(7,7))
for i in range(49):
    plt.subplot(7,7,i+1)
    plt.axis('off')
    plt.imshow(x_data[0][i], cmap='gray')

plt.show()
