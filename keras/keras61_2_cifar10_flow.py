'''
훈련데이터 10만개로 증폭 (기존데이터+ )
완료후 기존 모델과 비교
save_dir도 temp에 넣을것 - 들어가는것만 확인하고 지울것
증폭데이터 temp에 저장 후 훈련 끝나고 삭제
'''


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import cifar10
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=False,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=10,
    zoom_range=0.2,
    shear_range=0.5,
    fill_mode='nearest'
)

augment_size = 50000

randidx = np.random.randint(x_train.shape[0], size=augment_size)

x_argmented = x_train[randidx].copy()
y_argmented = y_train[randidx].copy()

x_argmented = x_argmented.reshape(x_argmented.shape[0], 32, 32, 3) # (40000, 28, 28, 1)
x_train = x_train.reshape(x_train.shape[0], 32, 32, 3) # (60000, 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 32, 32, 3) # (10000, 28, 28, 1)

x_argmented = train_datagen.flow(x_argmented, 
                                np.zeros(augment_size),
                                batch_size=augment_size,
                                # save_to_dir='d:/bitcamp/temp/',
                                shuffle=False).next()[0]

x_train = np.concatenate((x_train, x_argmented)) 
y_train = np.concatenate((y_train, y_argmented)) 

x_train = x_train.reshape(100000, 32*32*3)
x_test = x_test.reshape(10000, 32*32*3) 

from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) 

x_train = x_train.reshape(100000, 32*32, 3) 
x_test = x_test.reshape(10000, 32*32, 3) 


from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
one.fit(y_train)
y_train = one.transform(y_train).toarray()
y_test = one.transform(y_test).toarray() 



# 모델

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPool1D, GlobalAveragePooling1D, Dropout

model = Sequential()
model.add(Conv1D(filters=128, kernel_size=2, padding='same',                        
                        activation='relu' ,input_shape=(32*32, 3))) 
model.add(Conv1D(64, 2, padding='same', activation='relu'))                   
model.add(MaxPool1D())                                         
model.add(Conv1D(64, 2, padding='same', activation='relu'))                     
model.add(Flatten())                                              
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))


# 훈련

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)

import time 

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=576, verbose=2,
    validation_split=0.05, callbacks=[es])
end_time = time.time() - start_time



# 평가

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

loss = model.evaluate(x_test, y_test)
print('acc : ',acc[-10])
print('val_acc : ',val_acc[-10])
# print('loss : ',loss[-10])
print('val_loss : ',val_loss[-10])


'''
acc :  0.35223159193992615
val_acc :  0.09939999878406525
val_loss :  2.302900791168213
'''