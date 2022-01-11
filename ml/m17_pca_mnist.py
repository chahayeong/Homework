import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA


(x_train, _), (x_test, _) = mnist.load_data()
# print(x_train.shape, x_test.shape)          # (60000, 28, 28) (10000, 28, 28)

x = np.append(x_train, x_test, axis=0)
# print(x.shape)                              # (70000, 28, 28)

x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])

pca = PCA(n_components=500)              # 주성분을 몇개로 할지 결정
x = pca.fit_transform(x)
# print(x)
# print(x.shape)

pca_EVR = pca.explained_variance_ratio_

# print(pca_EVR)
print('sum : ',sum(pca_EVR))
cumsum = np.cumsum(pca_EVR)
# print(cumsum)
print('? : ', np.argmax(cumsum >= 1.0)+1)   # ()>= 분산비율) -> 결과값= 주성분(차원)



# pca 를 통해 0.95 이상의 n_components 가 몇개?
# 모델 구성


'''
# 0.95 : 154
# 0.99 : 331
# 1.0 : 1
'''