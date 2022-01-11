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



# 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten

model = Sequential()
# model.add(LSTM(64, 2, input_shape=(5,1)))
model.add(Conv1D(64, 2, input_shape=(5,1)))
model.add(LSTM(64, return_sequences=True))
model.add(Conv1D(64,2))         # LSTM 다음 Conv1D가 대체적으로 성능이 좋았음
model.add(Flatten())            #해주지 않으면 3차원으로 쭉 내려감
model.add(Dense(10))
model.add(Dense(1))

model.summary()