# tf09 1번의 방식 3가지로 hypothesis 출력

import tensorflow as tf
tf.compat.v1.set_random_seed(777)

x = [1,2,3]
W = tf.Variable([0.3], tf.float32)
b = tf.Variable([1.0], tf.float32)

hypothesis = x * W + b

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())
aaa = sess.run(hypothesis)
print("aaa : ", aaa)    # aaa : aaa :  [2.2086694]
sess.close()                    # 세션 닫힘

sess = tf.InteractiveSession()  # 세션을 열어줌
sess.run(tf.global_variables_initializer())
bbb = hypothesis.eval()      # 변수.eval
print("bbb : ", bbb)
sess.close()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
ccc = hypothesis.eval(session=sess)
print("ccc : ", ccc)
sess.close()
