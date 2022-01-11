from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.datasets import load_boston

# 1-1. data
datasets = load_boston()

# 1. data

x_data = datasets.data
y_data = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
      test_size=0.15, shuffle=True, random_state=1234)

# 2. model
# model = GradientBoostingRegressor()
model = XGBRegressor()

# 3. fit
model.fit(x_train, y_train)

# 4. pred
acc = model.score(x_test, y_test)
print('acc : ', acc) 

print(model.feature_importances_)



import matplotlib.pyplot as plt
from xgboost import plot_importance

'''
def plot_feature_importance_dataset(model):
      n_features = datasets.data.shape[1]
      plt.barh(np.arange(n_features), model.feature_importances_,
            align='center')
      plt.yticks(np.arange(n_features), datasets.feature_names)
      plt.xlabel("Feature Importances")
      plt.ylabel("Features")
      plt.ylim(-1, n_features)

plot_feature_importance_dataset(model)
plt.show()

'''

plot_importance(model)
plt.show()