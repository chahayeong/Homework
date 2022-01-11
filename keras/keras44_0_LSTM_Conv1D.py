import numpy as np
x_data = np.array(range(1, 101))
x_predict = np.array(range(96, 105))



size = 6

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

dataset = split_x(x_data, size)

# print(dataset)

x = dataset[:, :5].reshape(95,5,1)
y = dataset[:, 5]

# print("x : \n", x)
# print("y : ", y)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y,
      test_size=0.2, shuffle=True, random_state=66)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)




# 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten

model = Sequential()
model.add(Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(5, 1)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

model.summary()


# 컴파일
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=10, mode='min', verbose=1)
import time 

start_time = time.time()
model.fit(x_train, y_train, epochs=150, batch_size=64,
        validation_split=0.1, callbacks=[es])
end_time = time.time() - start_time




from sklearn.metrics import r2_score, mean_squared_error
y_pred = model.predict(x_test)
print('y_pred : \n', y_pred) 
print("time : ", end_time)

def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))
rmse = RMSE(y_test, y_pred)
print('rmse score : ', rmse)

r2 = r2_score(y_test, y_pred)
print('R^2 score : ', r2)

result = model.predict(x_test)
print('predict :', result)