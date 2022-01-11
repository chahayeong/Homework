import numpy as np
import matplotlib.pyplot as plt

f = lambda x: x**2 - 4*x + 6
x = np.linspace(-1, 6, 100)
# print(x)
y = f(x)

# 그래프 그리기
plt.plot(x, y, 'k-')    # k- 선
plt.plot(2, 2, 'sk')    # sk 점
plt.grid()              # 모눈
plt.xlabel('x')
plt.ylabel('y')

plt.show()
