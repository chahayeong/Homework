import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D
# tf.set_random_seed(66)

tf.compat.v1.disable_eager_execution()
# print(tf.executing_eagerly())       # False
# print(tf.__version__)               # 1.14.0 -> 2.6.0


#1. 데이터

# from tensorflow.keras.datasets import mnist
from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

from tensorflow.keras.utils import to_categorical
# from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# print(x_train.shape, x_test.shape)          # (50000, 32, 32, 3) (10000, 32, 32, 3)

x_train = x_train.reshape(50000, 32, 32, 3).astype('float32')/255
x_test = x_test.reshape(10000, 32, 32, 3).astype('float32')/255


learning_rate = 0.0001

training_epochs = 15
batch_size = 100
total_batch = int(len(x_train)/batch_size)

x = tf.compat.v1.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.compat.v1.placeholder(tf.float32, [None, 10])



# 2. 모델

w1 = tf.compat.v1.get_variable('w1', shape=[3, 3, 3, 32])           
                               
L1 = tf.nn.conv2d(x, w1, strides=[1,1,1,1], padding='SAME')
L1 = tf.nn.relu(L1)
L1_maxpool = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# Layer2
# model = Sequential()
# model.add(Conv2D(filters=32, kernel_size=(3,3), strides=1,
#             padding='same', input_shape=(28,28,1), activation='relu'))
# model.add(MaxPool2D())

print(L1)               # (None, 32, 32, 32)
print(L1_maxpool)       # (None, 16, 16, 32)

# Layer2
w2 = tf.compat.v1.get_variable('w2', shape=[3, 3, 32, 64])
L2 = tf.nn.conv2d(L1_maxpool, w2, strides=[1,1,1,1], padding='SAME')
L2 = tf.nn.selu(L2)
L2_maxpool = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print(L2)               # (None, 16, 16, 64)
print(L2_maxpool)       # (None, 8, 8, 64)

# Layer3
w3 = tf.compat.v1.get_variable('w3', shape=[3, 3, 64, 128])
L3 = tf.nn.conv2d(L2_maxpool, w3, strides=[1,1,1,1], padding='SAME')
L3 = tf.nn.elu(L3)
L3_maxpool = tf.nn.max_pool(L3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print(L3)               # (None, 8, 8, 128)
print(L3_maxpool)       # (None, 4, 4, 128)

# Layer4
w4 = tf.compat.v1.get_variable('w4', shape=[2, 2, 128, 64], ) # initializer=tf.contrib.layers.xavier_initializer())
L4 = tf.nn.conv2d(L3_maxpool, w4, strides=[1,1,1,1], padding='VALID')
L4 = tf.nn.elu(L4)
L4_maxpool = tf.nn.max_pool(L4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print(L4)               # (None, 3, 3, 64)
print(L4_maxpool)       # (None, 2, 2, 64)


# Flatten
L_flat = tf.reshape(L4_maxpool, [-1, 2*2*64])
print("플래튼 : ", L_flat)        


# Layer5 DNN
w5 = tf.compat.v1.get_variable("w5", shape=[2*2*64, 64], ) # initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random.normal([64],name='b1'))
L5 = tf.matmul(L_flat, w5) + b5
L5 = tf.nn.selu(L5)
# L5 = tf.nn.dropout(L5, keep_prob=0.2)
print(L5)      

# Layer6 DNN
w6 = tf.compat.v1.get_variable("w6", shape=[64, 32], ) # initializer=tf.contrib.layers.xavier_initializer())      # L5의 shape-output인 64를 그대로 가져옴
b6 = tf.Variable(tf.random.normal([32],name='b2'))
L6 = tf.matmul(L5, w6) + b6
L6 = tf.nn.selu(L6)
# L6 = tf.nn.dropout(L6, keep_prob=0.2)
print(L6)      

# Layer7 Sotfmax
w7 = tf.compat.v1.get_variable("w7", shape=[32, 10])      # L5의 shape-output인 64를 그대로 가져옴
b7 = tf.Variable(tf.random.normal([10],name='b3'))
L7 = tf.matmul(L6, w7) + b7
hypothesis = tf.nn.softmax(L7)
print(hypothesis)      


# 3. 컴파일, 훈련

loss = tf.reduce_mean(-tf.reduce_sum(y*tf.math.log(hypothesis), axis=1))

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
    

for epoch in range(training_epochs):
    avg_loss = 0

    for i in range(total_batch):
        start = i * batch_size
        end = start + batch_size

        batch_x, batch_y = x_train[start:end], y_train[start:end]

        feed_dict = {x:batch_x, y:batch_y}

        batch_loss, _ = sess.run([loss, optimizer], feed_dict=feed_dict)

        avg_loss += batch_loss/total_batch

    print('Epoch : ', '%04d' %(epoch + 1), 'loss : {:.9f}'.format(avg_loss))

prediction = tf.equal(tf.compat.v1.arg_max(hypothesis,1), tf.compat.v1.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
print('ACC : ', sess.run(accuracy, feed_dict={x:x_test, y:y_test}))


'''
ACC :  0.6616
'''