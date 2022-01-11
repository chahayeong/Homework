import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score
from sklearn.datasets import load_breast_cancer


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



# 2. 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, Dropout, GlobalAveragePooling2D, MaxPool2D, LSTM

input = Input(shape=(30, 1))
model1 = LSTM(32, activation='relu')(input)
model2 = Dense(64, activation='relu')(model1)
model3 = Dense(64, activation='relu')(model2)
model4 = Dense(8, activation='relu')(model3)
output = Dense(1, activation='sigmoid')(model4)

model = Model(inputs=input, outputs=output)



# 3. 컴파일 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', \
    metrics=['mse', 'accuracy'])                                             # binary

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)       # petience 최저점 나오면 5번까지 참다가 멈춘다 

import time 
start_time = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=128, verbose=2,
    validation_split=0.2, callbacks=[es])
end_time = time.time() - start_time


hist = model.fit(x_train, y_train, epochs=100, batch_size=128, \
    validation_split=0.2, callbacks=[es])

end_time = time.time() - start_time


# 4. 평가, 예측

loss = model.evaluate(x_test, y_test)
print("걸린시간 : ", end_time)
print('loss : ', loss[0])
print('accuracy : ', loss[1])




'''
걸린시간 :  29.86165714263916
loss :  0.36229148507118225
accuracy :  0.07451191544532776
'''