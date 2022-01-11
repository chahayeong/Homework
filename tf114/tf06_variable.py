import tensorflow as tf
sess = tf.Session()

x = tf.Variable([2], dtype=tf.float32) # , name='test')

init = tf.global_variables_initializer()
sess.run(init)

print("프린트 x 나왔는지 확인 : ", sess.run(x))