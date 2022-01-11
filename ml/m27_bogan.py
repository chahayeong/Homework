# [1, np.nan, np.nan, 8, 10]


# 결측치 처리
# 1. 행삭제
# 2. 0 넣기 (특정값) -> [1, 0, 0, 8, 10]
# 3 앞에값              [1, 1, 1, 8, 10]
# 4 뒤에값              [1, 8, 8, 8, 10]
# 5 중위값              [1, 4.5, 5.6, 8, 10]
# 6 보간
# 7 모델링 - predict

from pandas import DataFrame, Series
from datetime import datetime
import numpy as np
import pandas as pd

datestrs = ['8/13/2021', '8/14/2021', '8/15/2021', '8/16/2021', '8/17/2021']
dates = pd.to_datetime(datestrs)
print(dates)
print(type(dates))
print("==========================")

ts = Series([1, np.nan, np.nan, 8, 10], index=dates)
print(ts)

ts_intp_linear = ts.interpolate()
print(ts_intp_linear)
