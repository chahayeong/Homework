import numpy as np

# 55_1 파일 먼저 실행

x_data = np.load('./_save/_npy/k55_boston_x_data.npy')
y_data = np.load('./_save/_npy/k55_boston_y_data.npy')

print(x_data)
print(y_data)
print(x_data.shape, y_data.shape)
