import tensorflow as tf
tf.compat.v1.set_random_seed(777)

W = tf.Variable(tf.random_normal([1]), name='weight')
print(W)
# <tf.Variable 'weight:0' shape=(1,) dtype=float32_ref>

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())
aaa = sess.run(W)
print("aaa : ", aaa)    # aaa : aaa :  [2.2086694]
sess.close()                    # 세션 닫힘

sess = tf.InteractiveSession()  # 세션을 열어줌
sess.run(tf.global_variables_initializer())
bbb = W.eval()      # 변수.eval
print("bbb : ", bbb)
sess.close()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
ccc = W.eval(session=sess)
print("ccc : ", ccc)
sess.close()


# # 세가지 표현방식
# aaa, bbb, ccc 셋 다 같은 값
