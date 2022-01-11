import numpy as np


# 1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,4,3,5,8,7,10,9,6])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(1000, input_dim=1))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))


# 3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam, Adagrad, Adamax, Adadelta
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam

optimizer = Nadam(lr=0.0001)

model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
model.fit(x, y, epochs=100, batch_size=1)


# 4. 평가 예측
loss, mse = model.evaluate(x,y,batch_size=1)
y_pred = model.predict([11])

print('loss : ', loss, '결과물 : ', y_pred)



# Adam 0.01
# loss :  2.7160401344299316 결과물 :  [[8.976708]]
# Adam 0.001
# loss :  2.4414525032043457 결과물 :  [[9.679504]]
# Adam 0.0001
# loss :  2.484745740890503 결과물 :  [[10.770122]]

# Adagrad 0.01
# loss :  2.397099256515503 결과물 :  [[10.227737]]
# Adagrad 0.001
# loss :  2.4339098930358887 결과물 :  [[10.229112]]
# Adagrad 0.0001
# loss :  2.49641752243042 결과물 :  [[10.528047]]

# Adamax 0.01
# loss :  2.409372568130493 결과물 :  [[9.887011]]
# Adamax 0.001
# loss :  2.462334156036377 결과물 :  [[10.569497]]
# Adamax 0.0001
# loss :  2.4426398277282715 결과물 :  [[10.50458]]

# Adadelta 0.01
# loss :  2.4891037940979004 결과물 :  [[10.628046]]
# Adadelta 0.001
# loss :  6.635985374450684 결과물 :  [[6.968925]]
# Adadelta 0.0001
# loss :  30.19351577758789 결과물 :  [[1.302179]]

# RMSprop 0.01
# loss :  4.584437370300293 결과물 :  [[6.9375496]]
# RMSprop 0.001
# loss :  2.482820510864258 결과물 :  [[10.741213]]
# RMSprop 0.0001
# loss :  2.4488139152526855 결과물 :  [[10.136604]]

# SGD 0.01
# loss :  nan 결과물 :  [[nan]]
# SGD 0.001
# loss :  2.4293479919433594 결과물 :  [[10.008772]]
# SGD 0.0001
# loss :  2.4925296306610107 결과물 :  [[9.934179]]

# Nadam 0.01
# loss :  2.736633777618408 결과물 :  [[8.914545]]
# Nadam 0.001
# loss :  2.53391170501709 결과물 :  [[9.362017]]
# Nadam 0.0001
# loss :  2.4111921787261963 결과물 :  [[10.334675]]