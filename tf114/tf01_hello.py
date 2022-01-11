import tensorflow as tf
print(tf.__version__)

# print('hello world')

hello = tf.constant("Hello world")
print(hello)

# Tensor("Const:0", shape=(), dtype=string)

sess = tf.Session()
print("-------------------------------")
print(sess.run(hello))
print("-------------------------------")
# b'Hello World