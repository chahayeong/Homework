# ft08_2 파일의 lr을 수정해서
# epoch가 2000번이 아니라 100번 이하로 줄임
# 결과치는
# step=100 이하, w=1.9999, b=0.9999

import tensorflow as tf
tf.compat.v1.set_random_seed(66)

# x_train = [1,2,3]
# y_train = [1,2,3]

x_train = tf.compat.v1.placeholder(tf.float32, shape=[None])
y_train = tf.compat.v1.placeholder(tf.float32, shape=[None])

# W = tf.Variable(1, dtype = tf.float32)  # 랜덤하게 내맘대로 넣어준
# b = tf.Variable(1, dtype = tf.float32)  # 초기값

W = tf.Variable(tf.compat.v1.random_normal([1]), dtype = tf.float32) 
b = tf.Variable(tf.compat.v1.random_normal([1]), dtype = tf.float32) 

hypothesis = x_train * W + b

loss = tf.reduce_mean(tf.square(hypothesis - y_train))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.173)
train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(1, 81):
    # sess.run(train)
    _, loss_val, W_val, b_val = sess.run([train, loss, W, b],
                 feed_dict={x_train:[1,2,3], y_train:[1,2,3]})


    if step % 20 == 0:
        print(step, loss_val, W_val, b_val) # sess.run(loss), sess.run(W), sess.run(b))


'''

20 0.27943766 [0.67292506] [0.24676874]
40 0.012045603 [0.9045686] [0.125064]
60 0.00086678076 [0.96740985] [0.05709293]
80 0.00011279385 [0.9875841] [0.02508195]

'''
