import numpy as np
from sklearn.datasets import load_boston
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

datasets = load_boston()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)

'''
pca = PCA(n_components=7)
x2 = pca.fit_transform(x)
# print(x2)
# print(x2.shape)
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


# xgb 결과 : 0.999982906820517
# pca 결과 : 0.999982906820517
