from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.datasets import mnist

from sklearn.decomposition import PCA

(x_train, y_train), (x_test, y_test) = mnist.load_data() # (60000, 28, 28) (60000,) (10000, 28, 28) (10000,)

x = np.append(x_train, x_test, axis=0) # (70000, 28, 28)
y = np.append(y_train, y_test, axis=0) # (70000,)

x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])

pca = PCA(n_components=154)
x = pca.fit_transform(x)

pca_EVR = pca.explained_variance_ratio_

cumsum = np.cumsum(pca_EVR)


# print(np.argmax(cumsum >= 0.95)+1) 

x_train, x_test, y_train, y_test = train_test_split(x, y,
      test_size=0.14, shuffle=True, random_state=77)

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
model.add(Dense(128, activation='relu', input_shape=(154,)))
model.add(Dropout(0.2))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))



# 3. 컴파일

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)

import time

start_time = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=256, verbose=1,
    validation_split=0.25, callbacks=[es])
end_time = time.time() - start_time


# 4. 평가

loss = model.evaluate(x_test, y_test)
print("time : ", end_time)
print('loss : ', loss[0])
print('acc : ', loss[1])

'''
CNN
time =  12.263086080551147
loss :  0.08342713862657547
acc :  0.9821000099182129
DNN
time :  20.324632167816162
loss :  0.09825558215379715
acc :  0.9785000085830688
PCA_DNN
time :  39.4516875743866
loss :  0.13225620985031128
acc :  0.9647995233535767
'''


# tensorflow DNN으로 구성하고 기존 tensorflow DNN과 비교
# 기존 보다 더 좋게 만들기
# Y값 만들기