# mnist 데이터를 pca를 통해 cnn으로 구성
# (28, 28) -> 784 -> 차원축소 (625) -> (25, 25) -> CNN 모델 구성
# 차원 축소는 임의 설정

import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

(x_train, y_train), (x_test, y_test) = mnist.load_data()


x = np.append(x_train, x_test, axis=0) 
y = np.append(y_train, y_test, axis=0) 

x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])

pca = PCA(n_components=625)
x = pca.fit_transform(x)

pca_EVR = pca.explained_variance_ratio_

cumsum = np.cumsum(pca_EVR)


# print(np.argmax(cumsum >= 0.95)+1) 

x_train, x_test, y_train, y_test = train_test_split(x, y,
      train_size=0.8, shuffle=True, random_state=6)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = QuantileTransformer()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) 

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
one.fit(y_train)
y_train = one.transform(y_train).toarray() 
y_test = one.transform(y_test).toarray() 


# 2. 모델


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, GlobalAveragePooling2D, Dropout

model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(625,)))
model.add(Dense(128, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))



# 3. 컴파일

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)

import time

start_time = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=512, verbose=1,
    validation_split=0.25, callbacks=[es])
end_time = time.time() - start_time


# 4. 평가

loss = model.evaluate(x_test, y_test)
print("time : ", end_time)
print('loss : ', loss[0])
print('acc : ', loss[1])



time :  17.23308300971985
loss :  0.3493141531944275
acc :  0.9131428599357605