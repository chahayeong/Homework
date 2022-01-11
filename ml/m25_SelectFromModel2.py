# 실습
# 데이타 DIABETS

# 1. 상단 모델에 그리드서치 또는 랜덤서치로 튜닝한 모델 구성
# 최적의 R2 값과 피처임포턴스 구하기

# 2. 위 스레드 값으로 SelctFromModel 돌려서 최적의 피처 개수 구하기

# 3. 위 피처 개수로 피처 개수를 조정한 뒤
# 그걸로 다시 랜덤 서치 그리드 서치해서
# 최적의 R2 구하기

# 1번값과 3번값 비교        # 0.47이상 만들기


from xgboost import XGBRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV

x, y = load_diabetes(return_X_y=True)
print(x.shape, y.shape)


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66
)


# 2. 모델

model = XGBRegressor(n_jobs=8)


# 3. 훈련

model.fit(x_train, y_train)


#. 4. 평가, 예측

score = model.score(x_test, y_test)
print("model.score : ", score)

thresholds = np.sort(model.feature_importances_)   # sort - 순서대로 정렬
print(thresholds)
'''
[0.02593722 0.03284872 0.03821947 0.04788675 0.05547737 0.06321313
 0.06597802 0.07382318 0.19681752 0.3997987 ]
'''




# 컬럼 낮은 순서대로 하나씩 삭제 후 평가

print("==================================================")
for thresh in thresholds:
    print(thresh)
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    # print(selection)

    select_x_train = selection.transform(x_train)   
    select_x_test = selection.transform(x_test)
    print(select_x_train.shape, select_x_test.shape)

    selection_model = XGBRegressor(n_jobs=-1)
    selection_model.fit(select_x_train, y_train)

    y_predict = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_predict)

    print("Thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1],
        score*100))

