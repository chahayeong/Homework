# k_neighbors=3

from imblearn.over_sampling import SMOTE
from sklearn.datasets import load_wine
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import time
import warnings
warnings.filterwarnings('ignore')

datasets = pd.read_csv('../_data/winequality-white.csv',
                    index_col=None, header=0, sep=';')

datasets = datasets.values

x = datasets[:, :11]
y = datasets[:, 11]
print(x.shape, y.shape)     # (4898, 11) (4898,)


print(pd.Series(y).value_counts())
# 6.0    2198
# 5.0    1457
# 7.0     880
# 8.0     175
# 4.0     163
# 3.0      20
# 9.0       5
print(y)
# [6. 6. 6. ... 6. 7. 6.]


x_new = x[:-30]
y_new = y[:-30]
print(x_new.shape, y_new.shape)         # (4868, 11) (4868,)
print(pd.Series(y_new).value_counts())
# 6.0    2181
# 5.0    1450
# 7.0     875
# 8.0     175
# 4.0     162
# 3.0      20
# 9.0       5


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.75, shuffle=True, random_state=66, stratify=y
)
print(pd.Series(y_train).value_counts())
# 6.0    1648
# 5.0    1093
# 7.0     660
# 8.0     131
# 4.0     122
# 3.0      15
# 9.0       4

model = XGBClassifier(n_jobs=-1)

model.fit(x_train, y_train, eval_metric='mlogloss')

score = model.score(x_test, y_test)
print("증폭전 model.score : ", score)       # 증폭전 model.score :  0.643265306122449


#################### smote 적용 ####################
print("=====================================================")

smote = SMOTE(random_state=66, k_neighbors=3)

x_smote_train, y_smote_train = smote.fit_resample(x_train, y_train)

# print(pd.Series(y_smote_train).value_counts())
print(x_smote_train.shape, y_smote_train.shape)     # (11536, 11) (11536,)

print("smote 전 : ", x_train.shape, y_train.shape)
print("smote 후 : ", x_smote_train.shape, y_smote_train.shape)
print('smote전 레이블 값 분포 : \n', pd.Series(y_train).value_counts())
print('smote후 레이블 값 분포 : \n', pd.Series(y_smote_train).value_counts())

model2 = XGBClassifier(n_jobs=-1)
model2.fit(x_smote_train, y_smote_train, eval_metric='mlogloss')

score = model2.score(x_test, y_test)
print("model2.score : ", score)     # model2.score :  0.6318367346938776

