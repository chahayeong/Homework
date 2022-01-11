import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
tf.set_random_seed(66)


datasets = load_iris()

x_data = datasets.data 
y_data = datasets.target 

y_data = y_data.reshape(-1,1)  

from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
encoder.fit(y_data)
y_data = encoder.transform(y_data).toarray()

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
      test_size=0.2, shuffle=True, random_state=12)


# print(x_train.shape, y_train.shape)         # (120, 4) (120, 1)
   

x = tf.compat.v1.placeholder(tf.float32, shape=(None, 4))
y = tf.compat.v1.placeholder(tf.float32, shape=(None, 3)) 


w = tf.Variable(tf.random.normal([4,1]), name='weight')
b = tf.Variable(tf.random.normal([1,3]), name='bias')

# hypothesis = tf.matmul(x, w) + b  # 이것만 쓰면 linear        
hypothesis = tf.nn.softmax(tf.matmul(x,w) + b)


loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis), axis=1))

# optimize = tf.train.GradientDEscentOptimeizer(learning_rate=0.01)
# train = optimizer.minimizer(cost)
# 두 코드를 합쳐서 아래 코드
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())   # 변수 초기화

    for epochs in range(2001):
        cost_val, _ = sess.run([loss, optimizer],
                feed_dict={x:x_train, y:y_train})         # 훈련할 값
        if epochs % 200 == 0:
            print(epochs, "loss : ", cost_val)
    


    # predict
    results = sess.run(hypothesis, feed_dict={x:x_test})
    print(results, sess.run(tf.argmax(results, 1)))



