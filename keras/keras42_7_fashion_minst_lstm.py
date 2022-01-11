
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import fashion_mnist


# 1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()



# 2. 전처리
from sklearn.preprocessing import OneHotEncoder
one_hot = OneHotEncoder()

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

y_train = one_hot.fit_transform(y_train).toarray()     
y_test = one_hot.fit_transform(y_test).toarray()         




# 3. 모델
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, Dropout, GlobalAveragePooling2D, MaxPool2D, LSTM

input = Input(shape=(28, 28))
model1 = LSTM(20, activation='relu')(input)
model2 = Dense(15, activation='relu')(model1)
model3 = Dense(15, activation='relu')(model2)
model4 = Dense(10, activation='relu')(model3)
output = Dense(10, activation='sigmoid')(model4)

model = Model(inputs=input, outputs=output)



# 4. 컴파일, 훈련       metrics=['acc']
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)

model.fit(x_train, y_train, epochs=50, batch_size=8, verbose=1,
    validation_split=0.0015, callbacks=[es])




# 5. 평가, 예측      
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])


'''
loss :  1.658949375152588
accuracy :  0.3084000051021576
'''