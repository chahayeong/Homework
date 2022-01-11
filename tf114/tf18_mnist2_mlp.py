# from tensorflow.keras.datasets import mnist
from keras.datasets import mnist
import tensorflow as tf


# 2. 모델 구성
x_train = tf.compat.v1.placeholder(tf.float32, shape=[None, 28*28])
y_train = tf.compat.v1.placeholder(tf.float32, shape=[None, 10]) 

# 아웃풋레이어
w = tf.Variable(tf.random.normal([28*28, 10]), name='weight')
b = tf.Variable(tf.random.normal([10]), name='bias')

# hypothesis = x * w + b
layer1 = tf.nn.relu(tf.matmul(x_train, w) + b)
layer2 = tf.nn.elu(tf.matmul(x_train, w) + b)
layer3 = tf.nn.selu(tf.matmul(x_train, w) + b)
layer4 = tf.nn.sigmoid(tf.matmul(x_train, w) + b)
layer = tf.nn.dropout(layer4, keep_prob=0.3)

cost = -tf.reduce_mean(y_train*tf.log(layer)+(1-y_train)*tf.log(1-layer))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-6)
train = optimizer.minimize(cost)