from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

# 1. 데이터
datasets = load_diabetes()
x_train, x_test, y_train, y_test = train_test_split(
    datasets.data, datasets.target, train_size=0.8, random_state=66
)


# 2. 모델
# model = DecisionTreeClassifier(max_depth=4)
# model = RandomForestClassifier()
# model = GradientBoostingClassifier()
model = XGBClassifier() 


# 3. 훈련
model.fit(x_train, y_train)


# 4. 평가, 예측
acc = model.score(x_test, y_test)
print('acc : ', acc)

print(model.feature_importances_)

'''
acc :  0.011235955056179775
[0.11253505 0.02559427 0.12290945 0.11231385 0.10641155 0.11701162 
 0.10838841 0.06203577 0.12106586 0.11173416]
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