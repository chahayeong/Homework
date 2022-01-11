import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

datasets = load_iris()
# print(datasets.keys())
# dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename']) 

# print(datasets.target_names)        # ['setosa' 'versicolor' 'virginica']

x = datasets.data
y = datasets.target
# print(x.shape, y.shape)     # (150, 4) (150,)

df = pd.DataFrame(x, columns=datasets.feature_names)
df = pd.DataFrame(x, columns=datasets['feature_names'])
print(df)


# y컬럼 추가
df['Target'] = y
print(df.head())

print("============================ 상관계수 히트 맵 ============================")
# print(df.corr())        # 상관계수가 높으면 1에 가까움

import matplotlib.pyplot as plt
import seaborn as sns

# 색 그래프로 보여줌
sns.set(font_scale=1.2)
sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)

plt.show()