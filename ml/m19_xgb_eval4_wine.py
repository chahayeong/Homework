
from xgboost import XGBRegressor, XGBRFRegressor
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# 1. 데이터

datasets = load_wine()
x = datasets['data']
y = datasets['target']
# print(x.shape, y.shape)       



x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=66, \
    test_size=0.7)

# 스케일링                    
scaler = MinMaxScaler()
scaler.fit(x_train)                              
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)                   


# 2. 모델

model = XGBRegressor(n_estimators=300, learning_rate=0.2, n_jobs=1)       

# 3. 훈련

model.fit(x_train, y_train, verbose=1, eval_metric=['rmse','mae','auc','logloss'], 
            eval_set=[(x_train, y_train), (x_test, y_test)]
)


# 4. 평가
results = model.score(x_test, y_test)
print("results : ", results)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2 : ", r2)


'''
results :  0.8843470591306747
r2 :  0.8843470591306747

GradientBoostingClassifier
acc :  0.9722222222222222
컬럼 삭제 후 acc :  0.9166666666666666
'''


print('======================================================')

hist = model.evals_result()
print(hist)


# eval_results 그래프 그리기
import matplotlib.pyplot as plt

epochs = len(hist['validation_0']['logloss'])
x_axis = range(0, epochs)

fig, ax = plt.subplots()
ax.plot(x_axis, hist['validation_0']['logloss'], label='Train')
ax.plot(x_axis, hist['validation_1']['logloss'], label='Test')
ax.legend()
plt.ylabel('Log Loss')
plt.title('XGBoost Log Loss')
# plt.show()

fig, ax = plt.subplots()
ax.plot(x_axis, hist['validation_0']['rmse'], label='Train')
ax.plot(x_axis, hist['validation_1']['rmse'], label='Test')
ax.legend()
plt.ylabel('Rmse')
plt.title('XGBoost RMSE')
plt.show()

