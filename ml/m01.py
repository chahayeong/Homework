import numpy as np
from sklearn.datasets import load_iris
from sklearn.datasets import load_boston
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split        # train data, test data 분리
from sklearn.metrics import r2_score



# 1. 데이터 분석

datasets = load_iris()

x = datasets.data
y = datasets.target

print(x.shape, y.shape) # (150, 4) (150,)
print(y)


# 원핫인코딩 : one-hot-encoding
# 0 -> [1, 0, 0]
# 1 -> [0, 1, 0]
# 2 -> [0, 0, 1]

# [0, 1, 2, 1] -> 
# [[1, 0, 0]
# [0, 1, 0]
# [0, 0, 1]
# [0, 1, 0]]        (4,) -> (4, 3)

'''
# 원핫인코딩 : one-hot-encoding
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
'''

# train_test_split : train data와 test data 자동으로 분리 해주는 함수
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=66, \
    test_size=0.7)


from sklearn.preprocessing import MinMaxScaler, StandardScaler                        
scaler = StandardScaler()
scaler.fit(x_train)                                 # train만 학습하기
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)                    # test를 train에 반영하면 안됨




# 2. 모델 구성
'''
model = Sequential()                       
model.add(Dense(128, activation='relu', input_shape=(4,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))
'''

from sklearn.svm import LinearSVC

model = LinearSVC()






# 3. 컴파일, 훈련
'''
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=20, mode='min', verbose=1)       # petience 최저점 나오면 5번까지 참다가 멈춘다 

hist = model.fit(x_train, y_train, epochs=100, batch_size=8, \
    validation_split=0.2, callbacks=[es]) 
# validation_split은 val_loss
'''

model.fit(x_train, y_train)


# print(hist)                     
# # <tensorflow.python.keras.callbacks.History object at 0x0000021A11FAB070>



'''
print(hist.history.keys())          # dict_keys(['loss', 'val_loss'])
print("============== loss ==============")
print(hist.history['loss'])
print("============== val_loss ===============")
print(hist.history['val_loss'])

'''



print("============== 평가, 예측 ===============")
# 4. 예측 및 평가
results = model.score(x_test, y_test)
print(results)


'''
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])
'''

from sklearn.metrics import r2_score, accuracy_score

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print("accuracy_score : ", acc)




print("============== 예측 ===============")

print(y_test[:5])
y_predict2 = model.predict(x_test[:5])
print(y_predict2)










import matplotlib.pyplot as plt

'''
# 한글 깨짐 적용
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/NGULIM.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

plt.plot(hist.history['loss'])          # x: epoch / y: hist.history['loss']
plt.plot(hist.history['val_loss'])

plt.title("로스, 발_로스")
plt.xlabel('epochs')
plt.ylabel('loss, val_loss')
plt.legend(['train loss', 'val_loss'])
plt.show()
'''

print('small')