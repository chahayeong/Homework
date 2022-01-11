# 라벨 0 을 112개 삭제

from imblearn.over_sampling import SMOTE
from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import time
import warnings
from sklearn.metrics import accuracy_score, f1_score
warnings.filterwarnings('ignore')

datasets = load_breast_cancer()

x = datasets['data']
y = datasets['target']
print(x.shape, y.shape)   

print(pd.Series(y).value_counts())


######################################################################
#### 라벨 삭제
######################################################################

print("=======================================")

x = x[:-112]
y = y[:-112]

print(pd.Series(y).value_counts())      



x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.75, shuffle=True, random_state=66, stratify=y
)
print(pd.Series(y_train).value_counts())

model = XGBClassifier(n_jobs=-1)

model.fit(x_train, y_train, eval_metric='mlogloss')

score = model.score(x_test, y_test)
print("증폭전 model.score : ", score)      

y_pred = model.predict(x_test)
f1 = f1_score(y_test, y_pred, average='macro')
print("f1_score : ", f1)

#################### smote 적용 ####################
print("=====================================================")

smote = SMOTE(random_state=66, k_neighbors=60)

start_time = time.time()

x_smote_train, y_smote_train = smote.fit_resample(x_train, y_train)

end_time = time.time() - start_time

# print(pd.Series(y_smote_train).value_counts())
print(x_smote_train.shape, y_smote_train.shape)   

print("smote 전 : ", x_train.shape, y_train.shape)
print("smote 후 : ", x_smote_train.shape, y_smote_train.shape)
print('smote전 레이블 값 분포 : \n', pd.Series(y_train).value_counts())
print('smote후 레이블 값 분포 : \n', pd.Series(y_smote_train).value_counts())
print("SMOTE 경과시간 : ", end_time)

model2 = XGBClassifier(n_jobs=-1)
model2.fit(x_smote_train, y_smote_train, eval_metric='mlogloss')

score = model2.score(x_test, y_test)
print("model2.score : ", score)    

y_pred = model2.predict(x_test)
f1 = f1_score(y_test, y_pred, average='macro')
print("f1_score : ", f1)

