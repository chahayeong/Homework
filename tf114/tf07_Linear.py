import tensorflow as tf
tf.set_random_seed(66)

x_train = [1,2,3]
y_train = [1,2,3]

W = tf.Variable(1, dtype = tf.float32)  # 랜덤하게 내맘대로 넣어준
b = tf.Variable(1, dtype = tf.float32)  # 초기값

hypothesis = x_train * W + b

loss = tf.reduce_mean(tf.square(hypothesis - y_train))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(loss), sess.run(W), sess.run(b))

