from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
from tensorflow.python.keras.saving.save import load_model

# 1. 데이터

x1 = np.array([range(100), range(301, 401), range(1, 101)])
x2 = np.array([range(101, 201), range(411, 511), range(100, 200)])
x1 = np.transpose(x1) # (100,3)
x2 = np.transpose(x2) # (100,3)
y1 = np.array(range(1001, 1101)) # (100,)

x1_train, x1_test, x2_train, x2_test, y1_train, y1_test = train_test_split(x1, x2, y1,
      train_size=0.9, shuffle=True, random_state=10)

# 2. 모델
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input


# 모델 1
input1 = Input(shape=(3,))
dense1 = Dense(10, activation='relu', name='dense1')(input1)
dense2 = Dense(7, activation='relu', name='dense2')(dense1)
dense3 = Dense(5, activation='relu', name='dense3')(dense2)
output1 = Dense(11, name='output1')(dense3)

# 모델 2
input2 = Input(shape=(3,))
dense11 = Dense(10, activation='relu', name='dense11')(input2)
dense12 = Dense(10, activation='relu', name='dense12')(dense11)
dense13 = Dense(10, activation='relu', name='dense13')(dense12)
dense14 = Dense(10, activation='relu', name='dense14')(dense13)
output2 = Dense(12, name='output2')(dense14)


from tensorflow.keras.layers import concatenate

merge1 = concatenate([output1, output2]) 
merge2 = Dense(10, activation='relu')(merge1)
merge3 = Dense(5, activation='relu')(merge2)
last_output = Dense(1)(merge3)


model = Model(inputs=[input1, input2], outputs=last_output)

# model.summary()

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1,
            restore_best_weights=False)

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,
            save_best_only=True, filepath='./_save/ModelCheckPoint/keras49_MCP.h5')


model.fit([x1_train, x2_train], [y1_train], epochs=100, batch_size=4, verbose=1, 
            callbacks=[es, mcp], validation_split=0.2)

model.save('./_save/ModelCheckPoint/keras49_model_save.h5')

print('=================1. basic print=================')


# 4. 평가, 예측
y1_predict= model.predict([x1_test, x2_test])
# print(y1_predict)

loss = model.evaluate([x1_test, x2_test], y1_test)
print('loss : ', loss[0])

R2 = r2_score(y1_test, y1_predict)
print("R^2 score : ", R2)

print('=================2. load model=================')
model2 = load_model('./_save/ModelCheckPoint/keras49_model_save.h5')

y1_predict= model2.predict([x1_test, x2_test])
# print(y1_predict)

loss = model2.evaluate([x1_test, x2_test], y1_test)
print('loss : ', loss[0])

R2 = r2_score(y1_test, y1_predict)
print("R^2 score : ", R2)

print('============3. Model Check Point=============')
model3 = load_model('./_save/ModelCheckPoint/keras49_MCP.h5')

y1_predict= model3.predict([x1_test, x2_test])
# print(y1_predict)

loss = model3.evaluate([x1_test, x2_test], y1_test)
print('loss : ', loss[0])

R2 = r2_score(y1_test, y1_predict)
print("R^2 score : ", R2)




'''
restore_best_weights = False
=================1. basic print=================
1/1 [==============================] - 0s 91ms/step - loss: 0.5173 - mae: 0.6898
loss :  0.5173187851905823
R^2 score :  0.9990566073326428
=================2. load model=================
1/1 [==============================] - 0s 89ms/step - loss: 0.5173 - mae: 0.6898
loss :  0.5173187851905823
R^2 score :  0.9990566073326428
============3. Model Check Point=============
1/1 [==============================] - 0s 137ms/step - loss: 0.3304 - mae: 0.4714
loss :  0.33038634061813354
R^2 score :  0.9993975010595021



restore_best_weights = True  (성능 가장 좋은 모델 복구)
=================1. basic print=================
1/1 [==============================] - 0s 90ms/step - loss: 0.1113 - mae: 0.2880
loss :  0.11134111881256104
R^2 score :  0.9997969561535206
=================2. load model=================
1/1 [==============================] - 0s 92ms/step - loss: 0.1113 - mae: 0.2880
loss :  0.11134111881256104
R^2 score :  0.9997969561535206
============3. Model Check Point=============
1/1 [==============================] - 0s 90ms/step - loss: 0.1113 - mae: 0.2880
loss :  0.11134111881256104
R^2 score :  0.9997969561535206
'''