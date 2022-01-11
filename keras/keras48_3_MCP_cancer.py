import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score
from sklearn.datasets import load_breast_cancer
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.saving.save import load_model

# 이진분류 모델


datasets = load_breast_cancer()

# print(datasets.DESCR)
# print(datasets.feature_names)


# 1. 데이터 분석
x = datasets.data
y = datasets.target




x_train, x_test, y_train, y_test = train_test_split(x,y, shuffle=True, \
    random_state=66, test_size=0.7)


from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, PowerTransformer
scaler = MinMaxScaler()
scaler.fit(x_train) 
x_train = scaler.fit_transform(x_train)         # fit과 transfrom 같이 (train에서만)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1, 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1, 1)


'''
# 2. 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv1D, Flatten, Dropout, GlobalAveragePooling2D, MaxPool2D, LSTM, Conv1D, Flatten

model = Sequential()
model.add(Conv1D(64, 2, input_shape=(30,1)))
model.add(LSTM(64, return_sequences=True))
model.add(Conv1D(64,2))       
model.add(Flatten())          
model.add(Dense(10))
model.add(Dense(1))

model.summary()



# 3. 컴파일 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', \
    metrics=['mse', 'accuracy'])                                             # binary

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)       # petience 최저점 나오면 5번까지 참다가 멈춘다 
cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto',
                    filepath='./_save/ModelCheckPoint/keras48_3_MCP.hdf5')

import time 
start_time = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=128, verbose=2,
    validation_split=0.2, callbacks=[es])
end_time = time.time() - start_time


hist = model.fit(x_train, y_train, epochs=100, batch_size=128, \
    validation_split=0.2, callbacks=[es])

end_time = time.time() - start_time

model.save('./_save/ModelCheckPoint/keras48_3_MCP.hdf5')

'''

model = load_model('./_save/ModelCheckPoint/keras48_3_MCP.hdf5')

# 4. 평가, 예측

loss = model.evaluate(x_test, y_test)
# print("걸린시간 : ", end_time)
print('loss : ', loss[0])
print('accuracy : ', loss[1])




'''
============ 이전
걸린시간 :  29.86165714263916
loss :  0.36229148507118225
accuracy :  0.07451191544532776

============ Conv1D
걸린시간 :  7.430402994155884
loss :  0.17599798738956451
accuracy :  3.1990342140197754


============ MCP
loss :  0.1843184530735016
accuracy :  1.836142897605896
'''