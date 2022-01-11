import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)



pca = PCA(n_components=7)
x = pca.fit_transform(x)
# print(x)
print(x.shape)

pca_EVR = pca.explained_variance_ratio_

print(pca_EVR)
# [8.05823175e-01 1.63051968e-01 2.13486092e-02 6.95699061e-03 1.29995193e-03 7.27220158e-04 4.19044539e-04]
print(sum(pca_EVR))


cumsum = np.cumsum(pca_EVR)
# print(cumsum)

print(np.argmax(cumsum >= 0.94) + 1)

import matplotlib.pyplot as plt
plt.plot(cumsum)
plt.grid()
plt.show()



'''
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=66, shuffle=True
)


# 2. 모델
from xgboost import XGBRegressor
model = XGBRegressor()


# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
results = model.score(x_test,y_test)
print("결과 : ", results)
'''

'''
pca - 컬럼 축소
일정비율로 압축 됨 (삭제 아님, 차원 축소)
'''