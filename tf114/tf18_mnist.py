# from tensorflow.keras.datasets import mnist
from keras.datasets import mnist
import tensorflow as tf

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data() # (60000, 28, 28) (60000,) (10000, 28, 28) (10000,)

x_train = x_train.reshape(60000, 28*28*1)
x_test = x_test.reshape(10000, 28*28*1)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = QuantileTransformer()
# scaler.fit(x_train) 
# x_train = scaler.transform(x_train) 
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) 

x_train = x_train.reshape(60000, 28, 28)
x_test = x_test.reshape(10000, 28, 28)

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
one.fit(y_train)
y_train = one.transform(y_train).toarray() # (60000, 10)
y_test = one.transform(y_test).toarray() # (10000, 10)



# 2. 모델 구성
x_train = tf.compat.v1.placeholder(tf.float32, shape=[None, 28*28])
y_train = tf.compat.v1.placeholder(tf.float32, shape=[None, 10]) 

# 아웃풋레이어
w = tf.Variable(tf.random.normal([28*28, 10]), name='weight')
b = tf.Variable(tf.random.normal([10]), name='bias')

# hypothesis = x * w + b
hypothesis = tf.sigmoid(tf.matmul(x_train, w) + b)

cost = -tf.reduce_mean(y_train*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-6)
train = optimizer.minimize(cost)
