

from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.datasets import cifar100


# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

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
model.add(Dense(100, activation='softmax'))

# model.trainable=False   # 전체 모델 훈련을 동결


model2 = Sequential()
model2.add(vgg19)
model2.add(GlobalAveragePooling2D())
model2.add(Dense(100, activation='softmax'))




# 4. 컴파일, 훈련       metrics=['acc']
from tensorflow.keras.optimizers import Adam
optimizer = Adam(lr=0.1)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto', verbose=1, factor=0.5)

import time
start_time = time.time()

hist = model.fit(x_train, y_train, epochs=50, batch_size=1024, verbose=1,
    validation_split=0.2, callbacks=[es, reduce_lr])

end_time = time.time() - start_time

# 5. 평가, 예측      
loss = model.evaluate(x_test, y_test)
print("걸린시간: ", end_time)
print('loss[category] : ', loss[0])
print('loss[accuracy] : ', loss[1])



# 1. cifar 100
# trainable = True, FC : loss=?, acc=?
# 걸린시간:  229.5743260383606
# loss[category] :  nan
# loss[accuracy] :  0.009999999776482582
''''''''''''''''''''''''''''''''''''''''''''''''
# # trainable = True, GAP : loss=?, acc=?
# 걸린시간:  237.70159769058228
# loss[category] :  nan
# loss[accuracy] :  0.009999999776482582
''''''''''''''''''''''''''''''''''''''''''''''''
# # trainable = False, FC : loss=?, acc=?
# 걸린시간:  350.03151774406433
# loss[category] :  2.952348470687866
# loss[accuracy] :  0.32919999957084656
''''''''''''''''''''''''''''''''''''''''''''''''
# trainable = False, GAP : loss=?, acc=?
# 걸린시간:  370.816752910614
# loss[category] :  3.0486083030700684
# loss[accuracy] :  0.3280999958515167