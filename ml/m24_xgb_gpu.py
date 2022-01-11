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
# print(x.shape, y.shape)         # (506, 13) (506,)



x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=66, \
    test_size=0.7)

# 스케일링                    
scaler = MinMaxScaler()
scaler.fit(x_train)                              
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)                   


# 2. 모델

model = XGBRegressor(n_estimators=10000, learning_rate=0.01,
                    tree_method='gpu_hist',
                    predictor='gpu_predictor',
                    gpu_id=0)

# 3. 훈련
import time
start_time = time.time()
model.fit(x_train, y_train, verbose=1, eval_metric='rmse',      #'mae', 'logloss' 
            eval_set=[(x_train, y_train), (x_test, y_test)],
)
print("걸린시간: ", time.time() - start_time)

# njobs1: 걸린시간:  4.543560981750488
# njobs2: 걸린시간:  4.11553692817688
# njobs3: 걸린시간:  4.023393869400024
# njobs4: 걸린시간:  4.0978782176971436

# njobs-1 : 걸린시간:  5.825289011001587
# njobs8 : 걸린시간:  4.7702343463897705


'''
tree_method='gpu_hist',predictor='gpu_predictor',gpu_id=0)
걸린시간:  24.22123694419861
'''