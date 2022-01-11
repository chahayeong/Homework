from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

'''
(N, 5, 5, 1)
(N, 4, 4, 10)
(N, 3, 3, 20)
(N, 180)
'''
model = Sequential()                                                       
model.add(Conv2D(10, kernel_size=(2,2), padding='same', input_shape=(5, 5, 1)))        # 노드개수 10, kernel_size (2,2)
model.add(Conv2D(20, (2,2), activation='relu'))                        # (2,2)를 자를것이다.
model.add(Conv2D(30, (2,2))) 
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32))
model.add(Dense(1, activation='sigmoid'))

model.summary()


# 과제 1. Conv2D의 디폴트 엑티베이션
# 과제 2. cONV2D summary의 파라미터 갯수 완벽 이해