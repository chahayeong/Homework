from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# 1. 데이터
datasets = load_breast_cancer()
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
acc :  0.9649122807017544
[0.0528006  0.01101093 0.04424216 0.04952693 0.00598371 0.00584879 
 0.04840895 0.06195623 0.00358101 0.0038136  0.01524283 0.00441402 
 0.0061776  0.04109655 0.00351632 0.00545125 0.00606059 0.00702482 
 0.00460457 0.00428421 0.15193942 0.01742276 0.12959835 0.13133011 
 0.01459    0.0146904  0.02674728 0.11118552 0.00960936 0.00784112]
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