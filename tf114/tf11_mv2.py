import tensorflow as tf
tf.compat.v1.set_random_seed(66)

x_data = [[73, 51, 65],
         [92, 98, 11],
         [89, 31, 33],
         [99, 33, 100],
         [17, 66, 79]]
y_data = [[152], [185], [180], [205], [142]]
# x_data (5,3)  y_data (5,1) 

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1]) 

w = tf.Variable(tf.random.normal([3,1]), name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')

# hypothesis = x * w + b  # 행령의 연산은 이렇게 하면 안됨
hypothesis = tf.matmul(x, w) + b




cost = tf.reduce_mean(tf.square(hypothesis-y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)


sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

for epochs in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
            feed_dict={x:x_data, y:y_data})
    if epochs % 10 == 0:
        print(epochs, "cost : ", cost_val, "\n", hy_val)

sess.close()
