
from xgboost import XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# 1. 데이터

datasets = load_boston()
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

model = XGBRegressor(n_estimators=20, learning_rate=0.15, n_jobs=1)        # n_jobs 코어수

# 3. 훈련

model.fit(x_train, y_train, verbose=1, eval_metric='rmse',      #'mae', 'logloss' 
            eval_set=[(x_train, y_train), (x_test, y_test)],
            early_stopping_rounds=10
)


# 저장
import pickle
pickle.dump(model, open('./_save/_xgb_save/m23_pickle.dat','wb'))
