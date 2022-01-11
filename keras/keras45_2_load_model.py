import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar100, mnist


# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# 2. 전처리
# reshape 할 필요 없이 x_train = x_train/255. 만 해도 가능
# 4차원이 안되면 2차원으로 바꿔서 전처리
x_train = x_train.reshape(60000, 28 * 28)
x_test = x_test.reshape(10000, 28 * 28)       

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, PowerTransformer
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)         # fit과 transfrom 같이 (train에서만)
x_test = scaler.transform(x_test)


# 2차원으로 스케일링 하고 다시 4차원으로 원위치
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)


from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)





# 3. 모델
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout
from tensorflow.python.keras.layers.core import Flatten

'''
model = Sequential()
model.add(Conv2D(filters=128, kernel_size=(2, 2), padding='valid', activation='relu' ,input_shape=(28, 28, 1))) 
model.add(Dropout(0.2))
model.add(Conv2D(128, (2,2),padding='same', activation='relu'))                     
model.add(MaxPool2D()) 

model.add(Conv2D(128, (2,2),padding='valid', activation='relu')) 
model.add(Dropout(0.2))  
model.add(Conv2D(128, (2,2),padding='same', activation='relu'))  
model.add(MaxPool2D()) 

model.add(Conv2D(64, (2,2), activation='relu')) 
model.add(Dropout(0.2))
model.add(Conv2D(64, (2,2),padding='same', activation='relu')) 
model.add(MaxPool2D()) 

model.add(Flatten())                                              
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))
'''

model = load_model('./_save/keras45_1_save_model.h5')

model.summary()

# model.save('./_save/keras45_1_save_model.h5')




# 4. 컴파일, 훈련       metrics=['acc']
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=7, mode='min', verbose=1)

# 시간 걸어주기
import time
start_time = time.time()

# 메모리 터지면 batch_size 줄이기
hist = model.fit(x_train, y_train, epochs=5, batch_size=128, verbose=1,
    validation_split=0.2, callbacks=[es])

end_time = time.time() - start_time






# 5. 평가, 예측      
loss = model.evaluate(x_test, y_test)
print("걸린시간: ", end_time)
print('loss[category] : ', loss[0])
print('loss[accuracy] : ', loss[1])



'''
================ 이전
걸린시간:  1211.0673971176147
loss[category] :  2.123988628387451
loss[accuracy] :  0.4442000091075897

=========== save_model
걸린시간:  336.1303415298462
loss[category] :  0.025549858808517456
loss[accuracy] :  0.9934999942779541

=========== load_model
걸린시간:  95.47043561935425
loss[category] :  0.03278416767716408
loss[accuracy] :  0.9902999997138977
'''

