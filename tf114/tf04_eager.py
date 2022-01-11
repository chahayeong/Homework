import tensorflow as tf
print(tf.__version__)
print(tf.executing_eagerly())       # true

tf.compat.v1.disable_eager_execution()
    # 즉시 실행 모드? - tf1에서도 쓸수 있음

print(tf.executing_eagerly())       # false


# print('hello world')

hello = tf.constant("Hello world")
print(hello)
# Tensor("Const:0", shape=(), dtype=string)

# sess = tf.Session()
sess = tf.compat.v1.Session()
print(sess.run(hello))

# b'Hello World

