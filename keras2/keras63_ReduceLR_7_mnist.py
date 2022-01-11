
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape, y_train.shape)           # (60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape)             # (10000, 28, 28) (10000,)


# 2. 전처리
x_train = x_train.reshape(60000, 28 * 28 * 1)
x_test = x_test.reshape(10000, 28 * 28 * 1)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, PowerTransformer
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)         # fit과 transfrom 같이 (train에서만)
x_test = scaler.transform(x_test)
                  
# print(np.unique(y_train))                       # [0 1 2 3 4 5 6 7 8 9]

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)






# 3. 모델
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, Dropout, GlobalAveragePooling2D, MaxPool2D, LSTM

input = Input(shape=(28, 28))
model1 = LSTM(32, activation='relu')(input)
model2 = Dense(64, activation='relu')(model1)
model3 = Dense(64, activation='relu')(model2)
model4 = Dense(32, activation='relu')(model3)
output = Dense(10, activation='sigmoid')(model4)

model = Model(inputs=input, outputs=output)



# 4. 컴파일, 훈련       metrics=['acc']
from tensorflow.keras.optimizers import Adam
optimizer = Adam(lr=0.1)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto', verbose=1, factor=0.5)


# 시간 걸어주기
import time
start_time = time.time()

hist = model.fit(x_train, y_train, epochs=50, batch_size=128, verbose=1,
    validation_split=0.2, callbacks=[es, reduce_lr])

end_time = time.time() - start_time



# 5. 평가, 예측         predict 할 필요 없음 (acc로만 판단)
loss = model.evaluate(x_test, y_test)
print("걸린시간 : ", end_time)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

'''
걸린시간 :  247.1761920452118
loss :  0.0663791373372078
accuracy :  0.9819999933242798
'''

''' optimizer
걸린시간 :  112.96130585670471
loss :  2.3066844940185547
accuracy :  0.11349999904632568
'''