from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
import time


#1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],[1, 1.1, 1.2, 1.3, 1.4 ,1.5 ,1.6 ,1.5 ,1.4, 1.3],[10,9,8,7,6,5,4,3,2,1]]) # (3,10)


print(x.shape)                      # (3, 10)
x = np.transpose(x)                 #행과 열이 바뀜
print(x.shape)                      # (10, 3)

y = np.array([[11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])
print(y.shape)                      # (1, 10)
y = np.transpose(y)    
print(y.shape)                      # (10, 1)

x_pred = np.array([[10, 1.3, 1]])
print(x_pred.shape)                 # (1, 3)




# 2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=3))            # 결과값 나올 개수, input_dim = 차원






# 3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')


start = time.time()
# verbose 0일때 진행사항 안보여줌(시간절약), 1일땐 보여줌, 2일때 프로그래스(작업바)만 안보임
model.fit(x, y, epochs=1000, batch_size=1, verbose=2)           
end = time.time() - start
print("걸린시간 : ", end)

#verbose
# 0
# 걸린시간 :  0.6519355773925781 [epochs=1000, batch_size=10, verbose=0]
# 1
# 걸린시간 :  1.1544034481048584 [epochs=1000, batch_size=10, verbose=1]
# 2
# 걸린시간 :  0.9168055057525635 [epochs=1000, batch_size=10, verbose=2]
# 3
# 걸린시간 :  2.468665361404419 [epochs=1000, batch_size=1, verbose=0]
# 4
# 걸린시간 :  3.6270334720611572 [epochs=1000, batch_size=1, verbose=1]
# 5
# 걸린시간 :  2.864389419555664 [epochs=1000, batch_size=1, verbose=2]








loss = model.evaluate(x, y)
# print('loss : ', loss)

result = model.predict(x_pred)
# print('예측값: ', result)




y_predict = model.predict(x)



'''
x = np.transpose(x) 
plt.scatter(x[0],y)
plt.scatter(x[1],y)
plt.scatter(x[2],y)
x = np.transpose(x) 

plt.plot(x, y_predict, color='red')
plt.show()

'''