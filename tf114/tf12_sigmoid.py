import tensorflow as tf
tf.compat.v1.set_random_seed(66)

x_data = [[1,2], [2,3], [3,1], [4,3], [5,3], [6,2]]
y_data = [[0], [0], [0], [1], [1], [1]]

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1]) 

w = tf.Variable(tf.random.normal([2,1]), name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')


# hypothesis = x * w + b  # 행령의 연산은 이렇게 하면 안됨
hypothesis = tf.sigmoid(tf.matmul(x, w) + b)        # 예측값


# cost = tf.reduce_mean(tf.square(hypothesis-y))
cost = tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))


optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

# 3. 훈련
for epochs in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
            feed_dict={x:x_data, y:y_data})
    if epochs % 200 == 0:
        print(epochs, "cost : ", cost_val, "\n", hy_val)


# 4. 평가, 예측
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))


c, a = sess.run([predicted, accuracy], feed_dict={x:x_data, y:y_data})
print("==============================")
print("예측값 : ", hy_val, "\n 예측결과값 : ", c, "\n Accuracy : ", a)

sess.close()
