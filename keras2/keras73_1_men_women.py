# 가장 잘나온 전이학습 모델로
# 이 데이터를 학습시켜서 결과치 도출

from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16, VGG19

import numpy as np

# 1. 데이터
x_train = np.load('./_save/_npy/k59_5_train_x.npy')
x_test = np.load('./_save/_npy/k59_5_test_x.npy')
y_train = np.load('./_save/_npy/k59_5_train_y.npy')
y_test = np.load('./_save/_npy/k59_5_test_y.npy')

predict = np.load('./_save/_npy/k59_5_predict.npy')


print(x_train.shape, x_test.shape)      # (5, 150, 150, 3) (5, 150, 150, 3)


# 2. 모델

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, PowerTransformer
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

vgg19 = VGG19(weights='imagenet', include_top=False, input_shape=(150,150,3))
# model = VGG16()
# model = VGG19()

vgg19.trainable=True  

model = Sequential()
model.add(vgg19)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(1, activation='sigmoid'))



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



'''
이전
loss :  2.1611788272857666
acc:  0.800000011920929

걸린시간:  5.058523416519165
loss[category] :  nan
loss[accuracy] :  0.20000000298023224
'''