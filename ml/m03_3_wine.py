import numpy as np
from sklearn.datasets import load_wine

# 조건 : acc 0.8 이상 만들기


# 1. 데이터 분석

datasets = load_wine()

x = datasets.data
y = datasets.target




# print(x.shape, y.shape) # (178, 13) (178,)

'''
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)                           # one-hot coding : y값을 0과 1로만 이루어지게 만듦
'''

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=66, \
    test_size=0.7)


from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()
scaler.fit(x_train)                                
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) 


# 2. 모델 구성
'''
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()                       
model.add(Dense(128, activation='relu', input_shape=(13,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))
'''

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# model = LinearSVC()                   # accuracy_score :  0.992        
# model = KNeighborsClassifier()      # accuracy_score :  0.96
# model = LogisticRegression()        # accuracy_score :  0.984 
# model = DecisionTreeClassifier()    # accuracy_score :  0.864
model = RandomForestClassifier()    # accuracy_score :  0.976






# 3. 컴파일, 훈련
'''
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=20, mode='min', verbose=1)       # petience 최저점 나오면 5번까지 참다가 멈춘다 

hist = model.fit(x_train, y_train, epochs=1000, batch_size=8, \
    validation_split=0.2, callbacks=[es]) 
'''

model.fit(x_train, y_train)






# 4. 평가 및 예측
'''
print("============== 평가 ===============")

loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])


print("============== 예측 ===============")
print(y_test[:5])
y_predict = model.predict(x_test[:5])
print(y_predict)
'''

from sklearn.metrics import r2_score, accuracy_score

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print("accuracy_score : ", acc)