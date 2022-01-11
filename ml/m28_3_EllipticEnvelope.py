import numpy as np

aaa = np.array([[1   ,2   ,10000,3   ,4,   6,   7,   8,90  ,100 , 5000],
                [1000,2000,3    ,4000,5000,6000,7000,8,9000,10000,1001]])
# (2, 10) -> (10, 2)
aaa = aaa.transpose()
print(aaa.shape)

from sklearn.covariance import EllipticEnvelope

outliers = EllipticEnvelope(contamination=.2)
outliers.fit(aaa)

results = outliers.predict(aaa)

print(results)          # 이상치 위치 -1 표시

# 이상치 처리
# 1. 삭제
# 2. Nan 처리 후 -> 보간    // linear
# 3. ............ (결측치 처리 방법과 유사)
# 4. scaler -> Rubsorscaler, QuantileTransformer ... 등등등 ...
# 5. 모델링 : tree 계열... DT, RF, XG, LGBM

