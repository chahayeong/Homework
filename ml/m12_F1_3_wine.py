from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np


# 1. 데이터
datasets = load_wine()

x_data = datasets.data
y_data = datasets.target


df = pd.DataFrame(x_data, columns=datasets.feature_names)
x_data = df[['proline', 'od280/od315_of_diluted_wines','flavanoids']]




x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, train_size=0.8, random_state=66
)


# 2. 모델
model = DecisionTreeClassifier(max_depth=4)
# model = RandomForestClassifier()
# model = GradientBoostingClassifier()
# model = XGBClassifier() 


# 3. 훈련
model.fit(x_train, y_train)


# 4. 평가, 예측
acc = model.score(x_test, y_test)
print('acc : ', acc)

# print(model.feature_importances_)

'''
# 그래프
import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importances_dataset(model):
    n_features = datasets.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_,
    align='center')
    plt.yticks(np.arange(n_features), datasets.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)

plot_feature_importances_dataset(model)
plt.show()
'''


'''
DecisionTreeClassifier
acc :  0.9444444444444444
컬럼 삭제 후 acc :  0.9166666666666666

RandomForestClassifier
acc :  1.0
컬럼 삭제 후 acc :  0.9444444444444444

GradientBoostingClassifier
acc :  0.9722222222222222
컬럼 삭제 후 acc :  0.9166666666666666

XGBClassifier
acc :  1.0
컬럼 삭제 후 acc :  0.9722222222222222

'''