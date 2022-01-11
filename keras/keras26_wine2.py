# 다중분류
# 0.8 이상 완성
'''
1. 판다스 -> 넘파이
2. x와 y를 분리
3. sklearn의 onehot 사용
4. y의 라벨을 확인 np.unique(y)
5. y의 shape 확인 (4898,) -> (4898,7)
accuracy : 
'''


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

datasets = pd.read_csv('../_data/winequality-white.csv', sep=';', \
    index_col=None, header=0)

# print(datasets)
# print(datasets.shape)                                 # (4898, 12)
# print(datasets.info())
# print(datasets.describe())

datasets = datasets.to_numpy()                          # pandas를 numpy로 변경
datasets = np.transpose(datasets)

x = datasets[:11]
y = datasets[11]

x = np.transpose(x)                                     # (4898, 11)
y = np.transpose(y)                                     # (4898,)

y = y.reshape(-1,1)                                     # (?,) 1차원을 2차원으로 (?,?)


from sklearn.preprocessing import OneHotEncoder         # sklearn의 onehot 사용
onehot_encoder = OneHotEncoder()
y = onehot_encoder.fit_transform(y).toarray()           # 오류 난 부분

# print(y.shape)                                        # y (4898, 7)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=66, \
    test_size=0.7)


from sklearn.preprocessing import StandardScaler, RobustScaler  
scaler = RobustScaler()
scaler.fit(x_train)                                
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) 


# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(1024, activation='relu', input_dim=11))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(7, activation='softmax'))




# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='accuracy', patience=20, mode='min', verbose=1)       # petience 최저점 나오면 5번까지 참다가 멈춘다 

hist = model.fit(x_train, y_train, epochs=1000, batch_size=32, \
    validation_split=0.2, callbacks=[es]) 




# 4. 평가 및 예측

print("============== 평가 ===============")

loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

'''
print("============== 예측 ===============")
print(y_test[:5])
y_predict = model.predict(x_test[:5])
print(y_predict)
'''






