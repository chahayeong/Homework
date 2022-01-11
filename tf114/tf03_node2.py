# 덧셈
# 뺄셈
# 곱셈
# 나눗셈


import tensorflow as tf
node1 = tf.constant(2.0)
node2 = tf.constant(3.0)

node3 = tf.add(node1, node2)
node4 = tf.sub(node1, node2)
node5 = tf.mul(node1, node2)
node6 = tf.div(node1, node2)


sess = tf.Session()
print('sess.tun(node3) : ', sess.run(node3))
print('sess.tun(node4) : ', sess.run(node4))
print('sess.tun(node5) : ', sess.run(node5))
print('sess.tun(node6) : ', sess.run(node6))