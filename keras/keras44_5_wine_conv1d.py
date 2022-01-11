
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

datasets = pd.read_csv('../_data/winequality-white.csv', sep=';', \
    index_col=None, header=0)


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

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, PowerTransformer
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)    
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1, 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1, 1)


# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten

model = Sequential()
model.add(Conv1D(64, 2, input_shape=(11,1)))
model.add(LSTM(64, return_sequences=True))
model.add(Conv1D(64,2))       
model.add(Flatten())           
model.add(Dense(32))
model.add(Dense(7))

model.summary()




# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='accuracy', patience=20, mode='min', verbose=1)       # petience 최저점 나오면 5번까지 참다가 멈춘다 

import time 
start_time = time.time()

hist = model.fit(x_train, y_train, epochs=100, batch_size=8, \
    validation_split=0.2, callbacks=[es]) 

end_time = time.time() - start_time

# 5. 평가, 예측         predict 할 필요 없음 (acc로만 판단)
loss = model.evaluate(x_test, y_test)
print("걸린시간 : ", end_time)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

'''
======= 이전
걸린시간 :  76.65137195587158
loss :  1.1702646017074585
accuracy :  0.5048118829727173

====== Conv1D
걸린시간 :  16.145626068115234
loss :  11.995736122131348
accuracy :  0.44969379901885986

'''