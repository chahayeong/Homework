from re import X
import numpy as np
import tensorflow as tf
tf.set_random_seed(66)

# 원핫 인코딩이 되어있는 데이터
x_data = [[1,2,1,1],
         [2,1,3,2],
         [3,1,3,4],
         [4,1,5,5],
         [1,7,5,5],
         [1,2,5,6],
         [1,6,6,6],
         [1,7,6,7]]
y_data = [[0,0,1],
         [0,0,1],
         [0,0,1],
         [0,0,1],
         [0,0,1],
         [0,1,0],
         [0,1,0],
         [0,1,0],
         [1,0,0],
         [1,0,0]]

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 4])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 3]) 

# (None, 4) * w -> (N, 3) 으로 w 구함
#       (_)           (_) 
w = tf.Variable(tf.random.normal([4,3]), name='weight')
b = tf.Variable(tf.random.normal([1,3]), name='bias')

# hypothesis = tf.matmul(x, w) + b  # 이것만 쓰면 linear        
hypothesis = tf.nn.softmax(tf.matmul(x,w) + b)


loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis), axis=1))

# optimize = tf.train.GradientDEscentOptimeizer(learning_rate=0.01)
# train = optimizer.minimizer(cost)
# 두 코드를 합쳐서 아래 코드
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())   # 변수 초기화

    for epochs in range(2001):
        _, cost_val = sess.run([optimizer, loss],
                feed_dict={x:x_data, y:y_data})         # 훈련할 값
        if epochs % 200 == 0:
            print(epochs, "loss : ", cost_val)
    


# predict
results = sess.run(hypothesis, feed_dict={x:[[1, 11, 7, 9]]})
print(results, sess.run(tf.argmax(results, 1)))

sess.close()