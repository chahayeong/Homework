
#과제3
#보스턴 하우징 집값 Sequential()
from tensorflow.keras.models import Sequential
from sklearn.datasets import load_boston
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential, Model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split        # train data, test data 분리
from sklearn.metrics import r2_score



datasets = load_boston()



# 데이터 프레임으로 변환
df = pd.DataFrame(data=datasets.data, columns=datasets.feature_names)
df['price'] = datasets.target
# print(df)



# 1. 데이터 분석

 
x = df["LSTAT"]
y = df["price"]


# train_test_split : train data와 test data 자동으로 분리 해주는 함수
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)



# 모델 구성
input1 = Input(shape=(1,))
dense1 = Dense(10)(input1)
dense2 = Dense(5)(dense1)
dense3 = Dense(3)(dense2)
output1 = Dense(1)(dense3)

'''
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))
'''


# 3. 학습
model = Model(inputs=input1, outputs=output1)
model.summary()

model.compile(loss='mse', optimizer='adam')

model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=0, validation_split=0.2)


loss = model.evaluate(x_test, y_test)
print('loss : ', loss)


# 4. 예측 및 평가
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

print("r2 score : ", r2)

