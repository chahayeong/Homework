

# 1. [4]
# 2. [5,6]
# 3. [6,7,8]

# predict 하는 코드 추가
# x_test라는 placeholder 생성

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

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    # sess.run(train)
    _, loss_val, W_val, b_val = sess.run([train, loss, W, b],
                 feed_dict={x_train:[1,2,3], y_train:[1,2,3]})


    if step % 20 == 0:
        print(step, loss_val, W_val, b_val) # sess.run(loss), sess.run(W), sess.run(b))


x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])
predict_ex = x_test * W_val + b_val

predict1 = sess.run(predict_ex, feed_dict={x_test:[4]})
predict2 = sess.run(predict_ex, feed_dict={x_test:[5,6]})
predict3 = sess.run(predict_ex, feed_dict={x_test:[6,7,8]})

print(predict1)
print(predict2)
print(predict3)


# [3.9952912]
# [4.992564 5.989837]
# [5.989837 6.98711  7.984383]