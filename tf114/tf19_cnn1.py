import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D
tf.set_random_seed(66)

#1. 데이터

# from tensorflow.keras.datasets import mnist
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# from tensorflow.keras.utils import to_categorical
from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255

learning_rate = 0.001

training_epochs = 15
batch_size = 100
total_batch = int(len(x_train)/batch_size)

x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, 10])



# 2. 모델

w1 = tf.get_variable('w1', shape=[3, 3, 1, 32])             # 3, 3 (커널사이즈) 1 (placeholder의 1 (채널))
                                # [kernel_size (3,3), input, output]
L1 = tf.nn.conv2d(x, w1, strides=[1,1,1,1], padding='VALID')     # strides 1칸 간격으로 자르기, []개수는 차원맞추기 (실제는 가운데만 연산)

print(w1)   # (3,3,1,32)
print(L1)   # conv2d 연산 -> (?, 28, 28, 32)

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3,3), strides=1,
            padding='same', input_shape=(28,28,1)))

################### get_variable 연구 #######################
# w1 = tf.Variable(tf.random_normal([3, 3, 1, 32]), dtype = tf.float32)
# w1 = tf.Variable([1], dtype = tf.float32)

# .get_variable 시 초기값 자동 설정, 단, name과 shape 지정해줘야함
# .Variable 시 초기값 지정해줘야함

'''
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(np.min(sess.run(w1)))
print("=================================")
print(np.max(sess.run(w1)))
print("=================================")
print(np.mean(sess.run(w1)))
print("=================================")
print(np.median(sess.run(w1)))
print("=================================")
'''
############################################################

