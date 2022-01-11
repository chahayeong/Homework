from sklearn.datasets import load_diabetes
import tensorflow as tf
from sklearn.metrics import r2_score   

tf.set_random_seed(66)

datasets = load_diabetes()
x_data = datasets.data
y_data = datasets.target
print(x_data.shape, y_data.shape)     # (442, 10) (442,)

y_data = y_data.reshape(-1,1)


# 최종 결론 값은 r2_score로


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_data,y_data, train_size = 0.8,  random_state = 66)

x = tf.placeholder(tf.float32, shape=(None, 10))
y = tf.placeholder(tf.float32, shape=(None, 1))

w = tf.Variable(tf.random.normal([10,1]), name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')

hypothesis = tf.matmul(x, w) + b       


cost = tf.reduce_mean(tf.square(y-hypothesis))


optimizer = tf.train.AdamOptimizer(learning_rate=8e-1)
train = optimizer.minimize(cost)

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

# 3. 훈련
for epochs in range(5001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
            feed_dict={x:x_train, y:y_train})
    if epochs % 200 == 0:
        print(epochs, "cost : ", cost_val, "\n", hy_val)


# 4. 평가, 예측
predicted = sess.run(hypothesis, feed_dict={x:x_test})
r2 = r2_score(y_test, predicted)

print("r2 : ", r2)
sess.close()

'''
r2 :  0.504847440827392
'''