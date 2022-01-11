# cifar10 과 cifar100으로 모델 만들기
# trainable=True, False
# FC로 만든것과 GLOBAL AVARAGE POOLING으로 만든것 비교

# 결과 출력
# 1. cifar 10
# trainable = True, FC : loss=?, acc=?
# trainable = True, GAP : loss=?, acc=?
# trainable = False, FC : loss=?, acc=?
# trainable = False, GAP : loss=?, acc=?

# 1. cifar 100
# trainable = True, FC : loss=?, acc=?
# trainable = True, GAP : loss=?, acc=?
# trainable = False, FC : loss=?, acc=?
# trainable = False, GAP : loss=?, acc=?

from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.datasets import cifar10


# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.reshape(50000, 32 * 32 * 3)
x_test = x_test.reshape(10000, 32 * 32 * 3)       

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, PowerTransformer
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)        
x_test = scaler.transform(x_test)

x_train = x_train.reshape(50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32, 32, 3)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 2. 모델

vgg19 = VGG19(weights='imagenet', include_top=False, input_shape=(32,32,3))
# model = VGG16()
# model = VGG19()

vgg19.trainable=True   # vgg훈련을 동결

model = Sequential()
model.add(vgg19)
model.add(Flatten())
model.add(Dense(150))
model.add(Dense(10, activation='softmax'))

# model.trainable=False   # 전체 모델 훈련을 동결


model2 = Sequential()
model2.add(vgg19)
model2.add(GlobalAveragePooling2D())
model2.add(Dense(10, activation='softmax'))




# 4. 컴파일, 훈련       metrics=['acc']
from tensorflow.keras.optimizers import Adam
optimizer = Adam(lr=0.1)

model2.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto', verbose=1, factor=0.5)

import time
start_time = time.time()

hist = model2.fit(x_train, y_train, epochs=50, batch_size=1024, verbose=1,
    validation_split=0.2, callbacks=[es, reduce_lr])

end_time = time.time() - start_time

# 5. 평가, 예측      
loss = model2.evaluate(x_test, y_test)
print("걸린시간: ", end_time)
print('loss[category] : ', loss[0])
print('loss[accuracy] : ', loss[1])






# 1. cifar 10
# trainable = True, FC : loss=?, acc=?
# 걸린시간:  222.3225953578949
# loss[category] :  nan
# loss[accuracy] :  0.10000000149011612
''''''''''''''''''''''''''''''''''''''''''''''''''''''
# trainable = True, GAP : loss=?, acc=?
# 걸린시간:  214.87462615966797
# loss[category] :  nan
# loss[accuracy] :  0.10000000149011612
''''''''''''''''''''''''''''''''''''''''''''''''''''''
# trainable = False, FC : loss=?, acc=?
# 걸린시간:  313.7443926334381
# loss[category] :  1.2322460412979126
# loss[accuracy] :  0.578499972820282
''''''''''''''''''''''''''''''''''''''''''''''''''''''
# trainable = False, GAP : loss=?, acc=?
# 걸린시간:  195.83051252365112
# loss[category] :  1.2876532077789307
# loss[accuracy] :  0.5633999705314636